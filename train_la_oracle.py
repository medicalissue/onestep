"""
Lookahead Oracle Self-Distillation (LA-Oracle)
===============================================

Core Idea:
  Teacher 없이, lookahead 기반으로 oracle distribution을 정의하여 self-distillation.

Mathtical Framework:
  1. ΔH_pred ≈ -η⟨∇_z H, ∇_z CE⟩   (lookahead entropy change)
  2. H* = H(p) + clip(ΔH_pred, -κ_↓, κ_↑)   (oracle entropy)
  3. Δz = -η·(p - e_y)·(||h||² + 1)   (exact lookahead logits via last layer)
  4. z̃_oracle = z + Δz
  5. q_τ(c|x) ∝ exp(z̃_oracle_c / τ)
  6. Find τ* s.t. H(q_τ*) = H*   (bisection)
  7. q*(·|x) = q_τ*(·|x)   (final oracle distribution)
  8. L(θ) = CE + λ·KL(q* || p_θ)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import math
import time
import logging
import sys
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

from models import resnet20

# FAISS for fast k-NN (optional)
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

import random
import numpy as np

def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def entropy(probs):
    """Compute entropy of probability distributions. probs: (B, C) -> (B,)"""
    return -torch.sum(probs * torch.log(probs + 1e-8), dim=1)


class EntropyEMA:
    """
    EMA tracker for entropy statistics.

    Tracks:
    - H_min/H_max: Shannon entropy bounds (for H_target output range)
    - H_raw_min/H_raw_max: k-NN distance bounds (for normalization)

    This ensures H_target mapping uses consistent normalization across batches.
    """

    def __init__(self, decay=0.99, initial_H_min=0.5, initial_H_max=4.0):
        """
        Args:
            decay: EMA decay rate (0.99 = slow adaptation, 0.9 = fast)
            initial_H_min: initial minimum Shannon entropy estimate
            initial_H_max: initial maximum Shannon entropy estimate
        """
        self.decay = decay
        # Shannon entropy bounds (output range for H_target)
        self.H_min = initial_H_min
        self.H_max = initial_H_max
        # k-NN raw distance bounds (for normalization)
        self.H_raw_min = None
        self.H_raw_max = None
        self.initialized = False
        self.raw_initialized = False

    @torch.no_grad()
    def update(self, H_current):
        """
        Update Shannon entropy EMA statistics.

        Args:
            H_current: (B,) current Shannon entropy values
        """
        batch_min = H_current.min().item()
        batch_max = H_current.max().item()

        if not self.initialized:
            self.H_min = batch_min
            self.H_max = batch_max
            self.initialized = True
        else:
            self.H_min = self.decay * self.H_min + (1 - self.decay) * batch_min
            self.H_max = self.decay * self.H_max + (1 - self.decay) * batch_max

    @torch.no_grad()
    def update_raw(self, H_raw):
        """
        Update k-NN raw distance EMA statistics.

        Args:
            H_raw: (B,) raw k-NN based values (d * log(epsilon_k))
        """
        batch_min = H_raw.min().item()
        batch_max = H_raw.max().item()

        if not self.raw_initialized:
            self.H_raw_min = batch_min
            self.H_raw_max = batch_max
            self.raw_initialized = True
        else:
            self.H_raw_min = self.decay * self.H_raw_min + (1 - self.decay) * batch_min
            self.H_raw_max = self.decay * self.H_raw_max + (1 - self.decay) * batch_max

    def get_bounds(self):
        """Return current Shannon entropy EMA bounds (H_min, H_max)."""
        return self.H_min, self.H_max

    def get_raw_bounds(self):
        """Return current k-NN raw distance EMA bounds."""
        return self.H_raw_min, self.H_raw_max


class LogitMemoryBank:
    """
    Per-class FIFO queue for storing z_oracle logits.

    Enables same-class k-NN with sufficient samples regardless of batch size.
    Similar to MoCo/SimCLR memory bank approach.
    """

    def __init__(self, num_classes, queue_size_per_class, logit_dim, device='cuda'):
        """
        Args:
            num_classes: number of classes (e.g., 100 for CIFAR-100)
            queue_size_per_class: max samples to store per class
            logit_dim: dimension of logits (= num_classes)
            device: cuda or cpu
        """
        self.num_classes = num_classes
        self.queue_size = queue_size_per_class
        self.logit_dim = logit_dim
        self.device = device

        # Per-class queues: list of tensors
        # Each queue[c] is (current_size, logit_dim)
        self.queues = [torch.empty(0, logit_dim, device=device) for _ in range(num_classes)]

    @torch.no_grad()
    def update(self, z_oracle, targets):
        """
        Add new samples to their respective class queues.

        Args:
            z_oracle: (B, C) lookahead logits
            targets: (B,) class labels
        """
        z_oracle = z_oracle.detach()

        for c in range(self.num_classes):
            mask = (targets == c)
            if mask.sum() == 0:
                continue

            new_samples = z_oracle[mask]  # (n_c, C)

            # Concatenate new samples
            self.queues[c] = torch.cat([self.queues[c], new_samples], dim=0)

            # Trim to queue size (FIFO: keep most recent)
            if self.queues[c].size(0) > self.queue_size:
                self.queues[c] = self.queues[c][-self.queue_size:]

    @torch.no_grad()
    def get_class_samples(self, class_idx):
        """Get all stored samples for a given class."""
        return self.queues[class_idx]

    def get_stats(self):
        """Return statistics about memory bank."""
        sizes = [q.size(0) for q in self.queues]
        return {
            'total_samples': sum(sizes),
            'avg_per_class': sum(sizes) / self.num_classes,
            'min_per_class': min(sizes),
            'max_per_class': max(sizes),
            'num_empty': sum(1 for s in sizes if s == 0)
        }


class SubClusterBank:
    """
    클래스별 k개의 sub-cluster를 **logit space**에서 유지하여 local centroid 제공.

    핵심 아이디어:
    - 같은 클래스 내에서도 "늑대 비슷한 개" vs "고양이 비슷한 개" 구분
    - Logit space에서 가까운 샘플들의 centroid를 teacher로 사용
    - k-NN처럼 local하지만, centroid처럼 O(k) lookup으로 빠름

    왜 Logit space?
    - Feature collapse 방지 (feature는 학습 중 representation이 변할 수 있음)
    - Logit = W·h + b 이므로 feature 정보가 이미 포함됨
    - 기존 LogitMemoryBank와 일관된 공간 사용

    구조:
    - logit_centroids: (num_classes, k_clusters, logit_dim) - sub-cluster의 logit 중심
    - counts: (num_classes, k_clusters) - 각 sub-cluster에 할당된 샘플 수
    """

    def __init__(self, num_classes=100, k_clusters=4, logit_dim=100,
                 device='cuda', ema_decay=0.99):
        """
        Args:
            num_classes: 클래스 수 (e.g., 100 for CIFAR-100)
            k_clusters: 클래스당 sub-cluster 수
            logit_dim: logit 차원 (= num_classes)
            device: cuda or cpu
            ema_decay: EMA decay rate for centroid updates
        """
        self.num_classes = num_classes
        self.k_clusters = k_clusters
        self.logit_dim = logit_dim
        self.device = device
        self.ema_decay = ema_decay

        # Sub-cluster centroids in logit space
        # 초기화: 작은 랜덤 값 (학습 초기엔 의미없지만 빠르게 수렴)
        self.logit_centroids = torch.randn(num_classes, k_clusters, logit_dim, device=device) * 0.01

        # 각 sub-cluster에 할당된 샘플 수 (초기화 여부 판단용)
        self.counts = torch.zeros(num_classes, k_clusters, device=device)

        # Warmup: 최소 이만큼 샘플이 쌓여야 teacher로 사용
        self.min_samples_per_cluster = 5

    @torch.no_grad()
    def get_teacher_logits(self, logits, targets):
        """
        현재 logit과 가장 가까운 sub-cluster의 centroid 반환.

        Args:
            logits: (B, logit_dim) 현재 배치의 logit
            targets: (B,) ground truth labels

        Returns:
            teacher_logits: (B, logit_dim) 각 샘플의 teacher logit
            valid_mask: (B,) teacher가 유효한지 (충분히 학습됐는지)
        """
        B = logits.size(0)
        teacher_logits = torch.zeros(B, self.logit_dim, device=self.device)
        valid_mask = torch.zeros(B, dtype=torch.bool, device=self.device)

        for i in range(B):
            c = targets[i].item()

            # 해당 클래스의 k개 sub-cluster 중 logit과 가장 가까운 것 찾기
            # dists: (k_clusters,)
            dists = torch.norm(logits[i].unsqueeze(0) - self.logit_centroids[c], dim=1)
            nearest_idx = dists.argmin().item()

            # 해당 sub-cluster가 충분히 학습됐는지 확인
            if self.counts[c, nearest_idx] >= self.min_samples_per_cluster:
                teacher_logits[i] = self.logit_centroids[c, nearest_idx]
                valid_mask[i] = True
            else:
                # Fallback: 해당 클래스의 전체 평균 (있으면)
                total_count = self.counts[c].sum()
                if total_count >= self.min_samples_per_cluster:
                    # Weighted average of all sub-clusters
                    weights = self.counts[c] / (total_count + 1e-8)
                    teacher_logits[i] = (self.logit_centroids[c] * weights.unsqueeze(1)).sum(dim=0)
                    valid_mask[i] = True
                # else: valid_mask[i] = False (teacher 없음)

        return teacher_logits, valid_mask

    @torch.no_grad()
    def update(self, logits, targets):
        """
        EMA로 sub-cluster centroids 업데이트.

        각 샘플을 가장 가까운 sub-cluster에 할당하고 EMA 업데이트.

        Args:
            logits: (B, logit_dim) 현재 배치의 logit
            targets: (B,) ground truth labels
        """
        logits = logits.detach()

        for i in range(logits.size(0)):
            c = targets[i].item()

            # 가장 가까운 sub-cluster 찾기 (logit space에서)
            dists = torch.norm(logits[i].unsqueeze(0) - self.logit_centroids[c], dim=1)
            nearest_idx = dists.argmin().item()

            # EMA 업데이트
            if self.counts[c, nearest_idx] == 0:
                # 첫 샘플: 바로 할당
                self.logit_centroids[c, nearest_idx] = logits[i]
            else:
                # EMA
                self.logit_centroids[c, nearest_idx] = (
                    self.ema_decay * self.logit_centroids[c, nearest_idx] +
                    (1 - self.ema_decay) * logits[i]
                )

            self.counts[c, nearest_idx] += 1

    def get_stats(self):
        """Return statistics about sub-cluster bank."""
        total_clusters = self.num_classes * self.k_clusters
        active_clusters = (self.counts >= self.min_samples_per_cluster).sum().item()
        return {
            'total_clusters': total_clusters,
            'active_clusters': active_clusters,
            'active_ratio': active_clusters / total_clusters,
            'avg_samples_per_cluster': self.counts.sum().item() / total_clusters,
            'min_samples': self.counts.min().item(),
            'max_samples': self.counts.max().item(),
        }


def same_class_mean_distance(z, targets, memory_bank, eps=1e-10):
    """
    Memory bank 내 같은 클래스 샘플들과의 **평균 거리** 계산.

    k-NN 대신 전체 평균 사용:
    - 더 robust한 "클래스 중심으로부터의 거리" 추정
    - outlier 하나에 덜 민감
    - 클래스 내 분포 전체를 반영

    Args:
        z: (B, C) current batch logits
        targets: (B,) ground truth labels
        memory_bank: LogitMemoryBank instance
        eps: numerical stability

    Returns:
        H_raw: (B,) raw distance estimate (mean distance to same-class samples)
        mean_dists: (B,) mean distances
    """
    B, d = z.size()
    device = z.device

    mean_dists = torch.full((B,), float('inf'), device=device)
    unique_classes = targets.unique()

    for class_idx in unique_classes:
        class_idx = class_idx.item()
        batch_mask = (targets == class_idx)
        batch_indices = batch_mask.nonzero(as_tuple=True)[0]
        z_class = z[batch_mask]  # (n_batch, d)
        n_batch = z_class.size(0)

        if n_batch == 0:
            continue

        # Memory에서 같은 클래스 샘플 가져오기
        mem_samples = memory_bank.get_class_samples(class_idx)  # (M, d)
        n_mem = mem_samples.size(0)

        if n_mem > 0:
            # z_class와 memory의 모든 샘플 간 거리 계산
            # dist_matrix: (n_batch, M)
            dist_matrix = torch.cdist(z_class, mem_samples, p=2)
            # 평균 거리
            class_mean_dists = dist_matrix.mean(dim=1)  # (n_batch,)
            mean_dists[batch_indices] = class_mean_dists
        else:
            # Memory 비어있으면 batch 내에서 계산
            if n_batch > 1:
                dist_matrix = torch.cdist(z_class, z_class, p=2)
                # 자기 자신 제외
                mask = ~torch.eye(n_batch, dtype=torch.bool, device=device)
                class_mean_dists = dist_matrix[mask].view(n_batch, -1).mean(dim=1)
                mean_dists[batch_indices] = class_mean_dists

    mean_dists = torch.clamp(mean_dists, min=eps)

    # H_raw = d * log(mean_dist) - KL estimator 형태 유지
    H_raw = d * torch.log(mean_dists)

    return H_raw, mean_dists


def same_class_knn_entropy_faiss(z, targets, memory_bank, k=1, eps=1e-10):
    """
    FAISS 기반 same-class k-NN entropy 추정 (훨씬 빠름).

    NOTE: same_class_mean_distance()가 더 robust할 수 있음.

    Args:
        z: (B, C) current batch logits
        targets: (B,) ground truth labels
        memory_bank: LogitMemoryBank instance
        k: k-NN의 k값
        eps: numerical stability

    Returns:
        H_raw: (B,) raw entropy estimate
        epsilon_k: (B,) k-NN distances
    """
    B, d = z.size()
    device = z.device
    z_np = z.detach().cpu().numpy().astype('float32')

    epsilon_k = np.full(B, float('inf'), dtype='float32')
    unique_classes = targets.unique()

    for class_idx in unique_classes:
        class_idx = class_idx.item()
        batch_mask = (targets == class_idx).cpu().numpy()
        batch_indices = np.where(batch_mask)[0]
        z_class = z_np[batch_mask]  # (n_batch, d)
        n_batch = z_class.shape[0]

        if n_batch == 0:
            continue

        # Memory에서 같은 클래스 샘플 가져오기
        mem_samples = memory_bank.get_class_samples(class_idx)
        n_mem = mem_samples.size(0)

        if n_mem >= k:
            # FAISS index 생성 (L2 distance)
            mem_np = mem_samples.cpu().numpy().astype('float32')
            index = faiss.IndexFlatL2(d)
            index.add(mem_np)

            # k-NN search
            distances, _ = index.search(z_class, k)  # (n_batch, k)
            kth_dists = distances[:, k-1]  # k번째 거리 (0-indexed)
            # FAISS는 squared L2를 반환하므로 sqrt 필요
            kth_dists = np.sqrt(kth_dists)
            epsilon_k[batch_indices] = kth_dists
        else:
            # Memory 부족 시 batch + memory 결합
            if n_mem > 0:
                mem_np = mem_samples.cpu().numpy().astype('float32')
                combined = np.concatenate([mem_np, z_class], axis=0)
            else:
                combined = z_class

            n_combined = combined.shape[0]
            if n_combined > k:
                index = faiss.IndexFlatL2(d)
                index.add(combined)
                # k+1 search (자기 자신 포함될 수 있으므로)
                distances, indices = index.search(z_class, min(k + 1, n_combined))

                for i, idx in enumerate(range(n_batch)):
                    # 자기 자신 제외
                    self_pos = n_mem + i
                    valid_dists = []
                    for j in range(distances.shape[1]):
                        if indices[i, j] != self_pos:
                            valid_dists.append(distances[i, j])
                        if len(valid_dists) >= k:
                            break
                    if valid_dists:
                        epsilon_k[batch_indices[i]] = np.sqrt(valid_dists[k-1] if len(valid_dists) >= k else valid_dists[-1])

    epsilon_k = np.clip(epsilon_k, a_min=eps, a_max=None)
    H_raw = d * np.log(epsilon_k)

    return torch.tensor(H_raw, device=device), torch.tensor(epsilon_k, device=device)


def same_class_knn_entropy_with_memory(z, targets, memory_bank, k=1, eps=1e-10):
    """
    Memory bank 기반 same-class k-NN entropy 추정 (vectorized).

    Memory bank에서 같은 클래스 샘플들을 가져와 k-NN 계산.
    Batch size 제약 없이 충분한 same-class 샘플 확보 가능.

    Optimization tricks:
    - 클래스별 배치 처리로 vectorization
    - 전체 memory를 한번에 concat하고 클래스 마스킹

    Args:
        z: (B, C) current batch logits (lookahead logits z_oracle)
        targets: (B,) ground truth labels
        memory_bank: LogitMemoryBank instance
        k: k-NN의 k값 (default: 1)
        eps: numerical stability

    Returns:
        H_raw: (B,) raw entropy estimate (same-class k-NN distance)
        epsilon_k: (B,) k-NN distances
    """
    B, d = z.size()
    device = z.device

    # 배치 내 unique classes 찾기
    unique_classes = targets.unique()

    epsilon_k = torch.full((B,), float('inf'), device=device)

    for class_idx in unique_classes:
        class_idx = class_idx.item()
        batch_mask = (targets == class_idx)  # (B,) bool
        batch_indices = batch_mask.nonzero(as_tuple=True)[0]  # indices in batch
        z_class = z[batch_mask]  # (n_batch, d) - current batch samples of this class
        n_batch = z_class.size(0)

        if n_batch == 0:
            continue

        # Memory에서 같은 클래스 샘플 가져오기
        mem_samples = memory_bank.get_class_samples(class_idx)  # (M, d)
        n_mem = mem_samples.size(0)

        if n_mem >= k:
            # Memory bank에서 k-NN 계산 (vectorized)
            # z_class: (n_batch, d), mem_samples: (M, d)
            # dist_matrix: (n_batch, M)
            dist_matrix = torch.cdist(z_class, mem_samples, p=2)
            kth_dists, _ = torch.kthvalue(dist_matrix, k, dim=1)  # (n_batch,)
            epsilon_k[batch_indices] = kth_dists
        else:
            # Memory 부족 시 batch 내 같은 클래스 + memory 결합
            if n_mem > 0:
                combined = torch.cat([mem_samples, z_class], dim=0)  # (M+n_batch, d)
            else:
                combined = z_class  # (n_batch, d)

            n_combined = combined.size(0)
            if n_combined > k:
                # dist: (n_batch, M+n_batch)
                dist_matrix = torch.cdist(z_class, combined, p=2)
                # 자기 자신 제외 (combined의 마지막 n_batch개가 z_class)
                for i in range(n_batch):
                    dist_matrix[i, n_mem + i] = float('inf')
                kth_dists, _ = torch.kthvalue(dist_matrix, min(k, n_combined - 1), dim=1)
                epsilon_k[batch_indices] = kth_dists
            else:
                # 전체 batch로 fallback
                dist_all = torch.cdist(z_class, z, p=2)  # (n_batch, B)
                for i, idx in enumerate(batch_indices):
                    dist_all[i, idx] = float('inf')  # exclude self
                kth_dists, _ = torch.kthvalue(dist_all, min(k, B - 1), dim=1)
                epsilon_k[batch_indices] = kth_dists

    epsilon_k = torch.clamp(epsilon_k, min=eps)

    # H ∝ d * log(epsilon)
    H_raw = d * torch.log(epsilon_k)

    return H_raw, epsilon_k


def same_class_knn_entropy(z, targets, k=1, eps=1e-10):
    """
    Same-class k-NN distance 기반 entropy 추정 (batch-only, no memory bank).

    같은 클래스 샘플들과의 거리를 계산하여 difficulty 추정:
    - 같은 클래스 내에서 가까움 = 클래스 중심 = 쉬운 샘플 → lower H_target
    - 같은 클래스 내에서 멀음 = outlier = 어려운 샘플 → higher H_target

    Args:
        z: (B, C) logits (lookahead logits z_oracle)
        targets: (B,) ground truth labels
        k: k-NN의 k값 (default: 1)
        eps: numerical stability

    Returns:
        H_raw: (B,) raw entropy estimate (same-class k-NN distance)
        epsilon_k: (B,) k-NN distances
    """
    B, d = z.size()
    device = z.device

    # Pairwise L2 distances: (B, B)
    dist_matrix = torch.cdist(z, z, p=2)

    # Same-class mask: (B, B) - True if same class
    same_class_mask = (targets.unsqueeze(1) == targets.unsqueeze(0))  # (B, B)

    # Self-distance 제외 + 다른 클래스는 inf로
    dist_matrix = dist_matrix + torch.eye(B, device=device) * float('inf')
    dist_matrix = torch.where(same_class_mask, dist_matrix, torch.full_like(dist_matrix, float('inf')))

    # k-th nearest neighbor distance (within same class)
    # 같은 클래스 샘플이 k개 미만인 경우 처리 필요
    epsilon_k, _ = torch.kthvalue(dist_matrix, k, dim=1)  # (B,)

    # inf가 나온 경우 (같은 클래스 샘플이 k개 미만) → fallback to all-sample k-NN
    is_inf = torch.isinf(epsilon_k)
    if is_inf.any():
        # 전체 batch에서의 k-NN으로 fallback
        dist_all = torch.cdist(z, z, p=2)
        dist_all = dist_all + torch.eye(B, device=device) * float('inf')
        fallback_dist, _ = torch.kthvalue(dist_all, k, dim=1)
        epsilon_k = torch.where(is_inf, fallback_dist, epsilon_k)

    epsilon_k = torch.clamp(epsilon_k, min=eps)

    # H ∝ d * log(epsilon)
    # 같은 클래스 내에서 멀면 → epsilon 큼 → H_raw 큼 → 어려운 샘플
    H_raw = d * torch.log(epsilon_k)

    return H_raw, epsilon_k


def compute_centroid_oracle(logits, targets, memory_bank, blend_alpha=0.3):
    """
    Class centroid를 Oracle logit으로 사용.

    핵심 아이디어:
    - Memory bank의 같은 클래스 샘플들 평균 = "학습 완료 후 도달해야 할 위치"
    - Wrong samples에서 99.4% rank match (vs lookahead의 0.5%)
    - 즉, centroid가 "진정한 미래"를 더 잘 나타냄

    Args:
        logits: (B, C) current logits
        targets: (B,) ground truth labels
        memory_bank: LogitMemoryBank instance
        blend_alpha: 0=current logits, 1=pure centroid

    Returns:
        z_oracle: (B, C) oracle logits (blend of current and centroid)
        centroid_dist: (B,) distance to centroid (difficulty indicator)
    """
    B, C = logits.size()
    device = logits.device

    z_oracle = logits.clone()
    centroid_dist = torch.zeros(B, device=device)

    unique_classes = targets.unique()

    for class_idx in unique_classes:
        class_idx_item = class_idx.item()
        batch_mask = (targets == class_idx)
        batch_indices = batch_mask.nonzero(as_tuple=True)[0]

        # Get class centroid from memory bank
        mem_samples = memory_bank.get_class_samples(class_idx_item)
        n_mem = mem_samples.size(0)

        if n_mem > 0:
            centroid = mem_samples.mean(dim=0)  # (C,)

            # Blend: z_oracle = (1-α)·z + α·centroid
            for idx in batch_indices:
                z_oracle[idx] = (1 - blend_alpha) * logits[idx] + blend_alpha * centroid
                centroid_dist[idx] = torch.norm(logits[idx] - centroid)
        # else: keep original logits

    return z_oracle, centroid_dist


def compute_knn_oracle(logits, targets, memory_bank, k=5, blend_alpha=0.3):
    """
    k-NN weighted average를 Oracle logit으로 사용.

    Centroid보다 더 local한 정보 활용:
    - 현재 샘플과 가까운 k개의 같은 클래스 샘플들의 weighted average
    - Outlier에 덜 민감, 더 personalized target

    Args:
        logits: (B, C) current logits
        targets: (B,) ground truth labels
        memory_bank: LogitMemoryBank instance
        k: number of nearest neighbors
        blend_alpha: 0=current logits, 1=pure k-NN avg

    Returns:
        z_oracle: (B, C) oracle logits
        knn_dist: (B,) distance to k-th nearest neighbor
    """
    B, C = logits.size()
    device = logits.device

    z_oracle = logits.clone()
    knn_dist = torch.zeros(B, device=device)

    unique_classes = targets.unique()

    for class_idx in unique_classes:
        class_idx_item = class_idx.item()
        batch_mask = (targets == class_idx)
        batch_indices = batch_mask.nonzero(as_tuple=True)[0]

        mem_samples = memory_bank.get_class_samples(class_idx_item)
        n_mem = mem_samples.size(0)

        if n_mem >= k:
            for idx in batch_indices:
                # Distance to all memory samples
                dists = torch.norm(logits[idx:idx+1] - mem_samples, dim=1)  # (M,)

                # k nearest neighbors
                topk_dists, topk_idx = dists.topk(k, largest=False)
                topk_samples = mem_samples[topk_idx]  # (k, C)

                # Distance-weighted average (closer = higher weight)
                weights = 1.0 / (topk_dists + 1e-8)
                weights = weights / weights.sum()
                knn_avg = (topk_samples * weights.unsqueeze(1)).sum(dim=0)  # (C,)

                # Blend
                z_oracle[idx] = (1 - blend_alpha) * logits[idx] + blend_alpha * knn_avg
                knn_dist[idx] = topk_dists[-1]  # k-th distance
        elif n_mem > 0:
            # Fallback to centroid if not enough neighbors
            centroid = mem_samples.mean(dim=0)
            for idx in batch_indices:
                z_oracle[idx] = (1 - blend_alpha) * logits[idx] + blend_alpha * centroid
                knn_dist[idx] = torch.norm(logits[idx] - centroid)

    return z_oracle, knn_dist


def compute_adaptive_oracle(logits, probs, targets, memory_bank, k=5):
    """
    Confidence에 따라 adaptive하게 oracle 선택.

    핵심 통찰 (실험 결과):
    - High confidence: current logits와 oracle이 거의 동일 (이미 수렴)
    - Low confidence: centroid/k-NN이 훨씬 더 좋은 target

    전략:
    - High conf: 약한 blend (α 작음) - 미세 조정만
    - Low conf: 강한 blend (α 큼) - aggressive하게 centroid로 이동

    Args:
        logits: (B, C) current logits
        probs: (B, C) softmax probabilities
        targets: (B,) ground truth labels
        memory_bank: LogitMemoryBank instance
        k: k for k-NN average

    Returns:
        z_oracle: (B, C) adaptive oracle logits
        difficulty: (B,) difficulty score (used as blend_alpha)
    """
    B, C = logits.size()
    device = logits.device

    # Confidence = P(true class)
    confidence = probs[torch.arange(B, device=device), targets]

    # Adaptive alpha: low confidence → high alpha (more toward centroid)
    # α = 1 - confidence (range: [0, 1])
    # Clamp to reasonable range [0.1, 0.7]
    blend_alpha = torch.clamp(1.0 - confidence, min=0.1, max=0.7)

    z_oracle = logits.clone()
    difficulty = torch.zeros(B, device=device)

    unique_classes = targets.unique()

    for class_idx in unique_classes:
        class_idx_item = class_idx.item()
        batch_mask = (targets == class_idx)
        batch_indices = batch_mask.nonzero(as_tuple=True)[0]

        mem_samples = memory_bank.get_class_samples(class_idx_item)
        n_mem = mem_samples.size(0)

        if n_mem >= k:
            for idx in batch_indices:
                alpha = blend_alpha[idx].item()

                # k-NN weighted average
                dists = torch.norm(logits[idx:idx+1] - mem_samples, dim=1)
                topk_dists, topk_idx = dists.topk(k, largest=False)
                topk_samples = mem_samples[topk_idx]

                weights = 1.0 / (topk_dists + 1e-8)
                weights = weights / weights.sum()
                knn_avg = (topk_samples * weights.unsqueeze(1)).sum(dim=0)

                z_oracle[idx] = (1 - alpha) * logits[idx] + alpha * knn_avg
                difficulty[idx] = 1 - confidence[idx]
        elif n_mem > 0:
            centroid = mem_samples.mean(dim=0)
            for idx in batch_indices:
                alpha = blend_alpha[idx].item()
                z_oracle[idx] = (1 - alpha) * logits[idx] + alpha * centroid
                difficulty[idx] = 1 - confidence[idx]
        else:
            for idx in batch_indices:
                difficulty[idx] = 1 - confidence[idx]

    return z_oracle, difficulty


def compute_difficulty_score(H_raw, probs, targets, conf_weight=0.5):
    """
    k-NN distance와 confidence를 결합한 difficulty score 계산.

    핵심 통찰:
    - k-NN 높음 = 클래스 내 outlier (potentially hard)
    - confidence 낮음 = 모델이 실제로 어려워함 (actually hard)
    - 둘 다 높거나 낮을 때만 soft/sharp teacher가 의미있음

    Difficulty = (1 - conf_weight) * knn_difficulty + conf_weight * (1 - confidence)

    Cases:
    - k-NN 큼 + conf 낮음 → high difficulty → soft teacher ✓
    - k-NN 큼 + conf 높음 → medium difficulty → sharp teacher (기존에 놓침!)
    - k-NN 작음 + conf 높음 → low difficulty → sharp teacher ✓
    - k-NN 작음 + conf 낮음 → medium difficulty → investigate (이상 케이스)

    Args:
        H_raw: (B,) raw k-NN distance based entropy
        probs: (B, C) softmax probabilities
        targets: (B,) ground truth labels
        conf_weight: weight for confidence term (0=only k-NN, 1=only conf)

    Returns:
        difficulty: (B,) combined difficulty score in [0, 1]
    """
    B = H_raw.size(0)
    device = H_raw.device

    # 1. k-NN based difficulty (batch-relative normalization)
    H_min_batch = H_raw.min()
    H_max_batch = H_raw.max()
    h_range = H_max_batch - H_min_batch

    if h_range < 1e-6:
        knn_difficulty = torch.full_like(H_raw, 0.5)
    else:
        knn_difficulty = (H_raw - H_min_batch) / (h_range + 1e-8)  # [0, 1]

    # 2. Confidence based difficulty
    # confidence = P(true class)
    confidence = probs[torch.arange(B, device=device), targets]  # (B,)
    # high confidence → low difficulty
    conf_difficulty = 1.0 - confidence  # [0, 1]

    # 3. Combine (weighted average)
    difficulty = (1 - conf_weight) * knn_difficulty + conf_weight * conf_difficulty

    return difficulty, knn_difficulty, conf_difficulty


def map_kl_to_target_entropy(H_raw, H_min=0.0, H_max=None, C=100, entropy_ema=None,
                              use_batch_relative=True, probs=None, targets=None,
                              use_confidence=False, conf_weight=0.5):
    """
    Raw entropy를 Shannon entropy 범위 [H_min, H_max]로 매핑.

    Same-class k-NN의 경우 **정방향 매핑**:
    - H_raw 큼 (클래스 내 outlier) → H_target 큼 (어려운 샘플)
    - H_raw 작음 (클래스 중심) → H_target 작음 (쉬운 샘플)

    Collapse Prevention:
    - use_batch_relative=True: 항상 batch 내 상대적 순위로 H_target 결정
      → 학습 후반에도 H_target이 [H_min, H_max] 전체 범위 활용
    - use_batch_relative=False: Global EMA bounds 사용 (collapse 위험)

    Args:
        H_raw: (B,) raw entropy estimates (d * log(epsilon_k))
        H_min: minimum target entropy (default: 0, or from EMA)
        H_max: maximum target entropy (default: log(C), or from EMA)
        C: number of classes
        entropy_ema: EntropyEMA instance for adaptive bounds (optional)
        use_batch_relative: if True, always use batch-relative normalization
                           to prevent collapse (recommended)

    Returns:
        H_target: (B,) target entropy in [H_min, H_max]
    """
    # Output range: use EMA Shannon entropy bounds if available
    if entropy_ema is not None and entropy_ema.initialized:
        H_min, H_max = entropy_ema.get_bounds()
    else:
        # Fallback to sensible defaults when EMA not available
        if H_min is None:
            H_min = 0.5  # Reasonable minimum (not too deterministic)
        if H_max is None:
            H_max = math.log(C)

    # Normalization strategy:
    # - Batch-relative: 항상 배치 내 min=0, max=1로 정규화
    #   → 같은 샘플이라도 배치 구성에 따라 H_target 다름
    #   → 하지만 collapse 방지: 항상 전체 [H_min, H_max] 범위 사용
    # - Global EMA: EMA bounds로 정규화
    #   → 같은 샘플은 비슷한 H_target
    #   → 하지만 학습 후반 collapse 위험

    if use_batch_relative:
        # Always use batch-relative to prevent collapse
        # This ensures H_target always spans [H_min, H_max] within each batch
        H_min_batch = H_raw.min()
        H_max_batch = H_raw.max()
        h_range = H_max_batch - H_min_batch

        if h_range < 1e-6:
            # Edge case: all samples have same H_raw
            H_normalized = torch.full_like(H_raw, 0.5)
        else:
            H_normalized = (H_raw - H_min_batch) / (h_range + 1e-8)
    else:
        # Global EMA normalization (may collapse in late training)
        if entropy_ema is not None and entropy_ema.raw_initialized:
            H_raw_min, H_raw_max = entropy_ema.get_raw_bounds()
            H_raw_clamped = torch.clamp(H_raw, min=H_raw_min, max=H_raw_max)
            H_normalized = (H_raw_clamped - H_raw_min) / (H_raw_max - H_raw_min + 1e-8)
        else:
            H_min_batch = H_raw.min()
            H_max_batch = H_raw.max()
            H_normalized = (H_raw - H_min_batch) / (H_max_batch - H_min_batch + 1e-8)

    # **정방향 매핑**: H_raw 큼 → H_target 큼
    H_target = H_min + H_normalized * (H_max - H_min)

    return H_target


def compute_entropy_gradient(probs):
    """
    Compute gradient of entropy w.r.t. logits.
    
    ∂H/∂z_j = -p_j·(log p_j + H)
    
    Derivation:
      H = -Σ p_c log p_c
      ∂H/∂p_c = -log p_c - 1
      ∂p_c/∂z_j = p_c(δ_{cj} - p_j)
      ∂H/∂z_j = Σ_c (∂H/∂p_c)(∂p_c/∂z_j) = -p_j(log p_j + H)
    
    Args:
        probs: (B, C) softmax probabilities
    Returns:
        g_H: (B, C) gradient of H w.r.t. logits
        H: (B,) current entropy
    """
    log_probs = torch.log(probs + 1e-8)
    H = -torch.sum(probs * log_probs, dim=1, keepdim=True)  # (B, 1)
    g_H = -probs * (log_probs + H)  # (B, C)  -- FIXED: was probs * (log_probs + 1 + H)
    return g_H, H.squeeze(1)


def compute_ce_gradient(probs, targets):
    """
    Compute gradient of CE loss w.r.t. logits.
    
    ∂CE/∂z = p - e_y
    
    Args:
        probs: (B, C) softmax probabilities
        targets: (B,) ground truth indices
    Returns:
        g_CE: (B, C) gradient of CE w.r.t. logits
    """
    B, C = probs.size()
    one_hot = F.one_hot(targets, num_classes=C).float()
    g_CE = probs - one_hot  # (B, C)
    return g_CE


def compute_lookahead_entropy_change(probs, delta_z):
    """
    Compute predicted entropy change using actual Δz.
    
    ΔH_pred = ⟨∇_z H, Δz⟩
    
    This is consistent with the Δz computed by compute_lookahead_logits,
    including scale factor (||h||² + 1) and weight decay if enabled.
    
    Args:
        probs: (B, C) softmax probabilities
        delta_z: (B, C) predicted logit change
    Returns:
        delta_H: (B,) predicted entropy change per sample
        H_current: (B,) current entropy
    """
    g_H, H_current = compute_entropy_gradient(probs)
    
    # ΔH = ⟨∇_z H, Δz⟩
    delta_H = torch.sum(g_H * delta_z, dim=1)  # (B,)
    
    return delta_H, H_current


def compute_lookahead_logits(logits, probs, targets, features, eta=1.0, weight_decay=0.0,
                              fc_layer=None, optimizer=None, momentum=0.0):
    """
    Compute exact lookahead logits using analytical gradient for last linear layer.
    
    For z = W·h + b with SGD momentum and weight decay:
      v_W = μ·v_W_prev + g_W + λ_wd·W
      v_b = μ·v_b_prev + g_b + λ_wd·b
      
      W⁺ = W - η·v_W
      b⁺ = b - η·v_b
      
      z⁺ = z - η·(momentum_term + ce_term + wd_term)
      
    where:
      momentum_term = μ·(v_W_prev·h + v_b_prev)  (from optimizer state)
      ce_term = (p - e_y)·(||h||² + 1)  (current gradient contribution)
      wd_term = λ_wd·z  (weight decay shrinkage)
    
    Args:
        logits: (B, C) current logits
        probs: (B, C) softmax probabilities
        targets: (B,) ground truth indices
        features: (B, D) pre-fc features
        eta: learning rate (ODE time step)
        weight_decay: weight decay coefficient (0 to disable)
        fc_layer: nn.Linear layer (for momentum buffer access)
        optimizer: optimizer with state dict (for momentum buffer access)
        momentum: momentum coefficient (0 to disable)
    Returns:
        z_oracle: (B, C) lookahead oracle logits
        delta_z: (B, C) logit change
    """
    B, C = logits.size()
    device = logits.device
    
    # CE gradient w.r.t. logits: g_CE = p - e_y
    g_CE = compute_ce_gradient(probs, targets)  # (B, C)
    
    # Feature norm squared + 1 (for bias term)
    h_norm_sq = torch.sum(features ** 2, dim=1, keepdim=True)  # (B, 1)
    scale = h_norm_sq + 1.0  # (B, 1)
    
    # CE term: (p - e_y)·(||h||² + 1)
    ce_term = g_CE * scale  # (B, C)
    
    # Momentum term from optimizer state
    momentum_term = torch.zeros_like(logits)
    if momentum > 0 and fc_layer is not None and optimizer is not None:
        state_W = optimizer.state.get(fc_layer.weight, {})
        state_b = optimizer.state.get(fc_layer.bias, {})
        
        if 'momentum_buffer' in state_W and 'momentum_buffer' in state_b:
            v_W = state_W['momentum_buffer']  # (C, D)
            v_b = state_b['momentum_buffer']  # (C,)
            # momentum_term = μ·(v_W·h + v_b) for each sample
            momentum_term = momentum * (features @ v_W.T + v_b.unsqueeze(0))  # (B, C)
    
    # Weight decay term: λ_wd·z
    wd_term = weight_decay * logits  # (B, C)
    
    # Total Δz = -η·(momentum_term + ce_term + wd_term)
    delta_z = -eta * (momentum_term + ce_term + wd_term)
    
    # Oracle logits
    z_oracle = logits + delta_z
    
    return z_oracle, delta_z


def find_tau_star(z_oracle, H_target, tau_min=0.1, tau_max=10.0, n_iters=10):
    """
    Find temperature τ* such that H(softmax(z_oracle/τ*)) = H_target.
    
    Uses hybrid bisection + secant method for fast convergence.
    
    Args:
        z_oracle: (B, C) oracle logits
        H_target: (B,) target entropy
        tau_min, tau_max: search bounds
        n_iters: number of iterations
    Returns:
        tau_star: (B,) optimal temperature per sample
        q_star: (B, C) final oracle distribution
    """
    B, C = z_oracle.size()
    device = z_oracle.device
    
    tau_lo = torch.full((B,), tau_min, device=device)
    tau_hi = torch.full((B,), tau_max, device=device)
    tau = (tau_lo + tau_hi) / 2.0
    
    for _ in range(n_iters):
        tau_expanded = tau.view(-1, 1)
        
        # Forward: q_τ = softmax(z_oracle / τ)
        scaled_logits = z_oracle / tau_expanded
        q = F.softmax(scaled_logits, dim=1)
        H_cur = entropy(q)
        
        diff = H_cur - H_target
        
        # Bisection update
        # H is monotonically increasing with τ
        # If H > H_target (too flat) -> need smaller τ -> hi = τ
        # If H < H_target (too sharp) -> need larger τ -> lo = τ
        mask_too_flat = diff > 0
        tau_hi = torch.where(mask_too_flat, tau, tau_hi)
        tau_lo = torch.where(~mask_too_flat, tau, tau_lo)
        
        # Secant step (gradient-based acceleration)
        # dH/dτ = Var_q[z] / τ³
        # With z_scaled = z/τ: Var[z_scaled] = Var[z]/τ²
        # So: dH/dτ = Var[z_scaled] * τ² / τ³ = Var[z_scaled] / τ
        z_scaled = z_oracle / tau_expanded
        E_z = torch.sum(q * z_scaled, dim=1)
        E_z2 = torch.sum(q * z_scaled ** 2, dim=1)
        var_z_scaled = E_z2 - E_z ** 2  # = Var[z]/τ²
        dH_dtau = var_z_scaled / (tau + 1e-10)  # = Var[z]/τ³ ✓
        
        # Newton step: τ_new = τ - diff / dH_dtau
        tau_secant = tau - diff / (dH_dtau + 1e-10)
        
        # Only use secant if within bracket
        buffer = (tau_hi - tau_lo) * 0.1
        is_valid = (tau_secant > tau_lo + buffer) & (tau_secant < tau_hi - buffer)
        
        # Fallback to bisection
        tau_bisect = (tau_lo + tau_hi) / 2.0
        tau = torch.where(is_valid, tau_secant, tau_bisect)
    
    # Final distribution
    tau_star = tau
    q_star = F.softmax(z_oracle / tau_star.view(-1, 1), dim=1)
    
    return tau_star, q_star


def la_oracle_loss(logits, features, targets, cfg, current_lambda, current_lr,
                    fc_layer=None, optimizer=None, memory_bank=None, entropy_ema=None,
                    subcluster_bank=None):
    """
    Compute Lookahead Oracle Self-Distillation loss.

    ODE Interpretation:
      Oracle q* is the CE ODE solution integrated forward by time Δt = current_lr.
      This includes momentum buffer for accurate lookahead.

    L(θ) = mean_i[(1-λ_i)*CE_i + λ_i*KL_i]

    Sample-Adaptive λ (Batch-Relative Difficulty):
      d_i = |ΔH_pred,i| / batch_mean(|ΔH_pred|)
      λ_i = clamp(d_i / 2, 0, 0.5)

    Lookahead Δz includes:
      - momentum_term: μ·(v_W·h + v_b) from optimizer state
      - ce_term: (p - e_y)·(||h||² + 1) from current gradient
      - wd_term: λ_wd·z from weight decay

    Args:
        logits: (B, C) model logits
        features: (B, D) pre-fc features
        targets: (B,) ground truth indices
        cfg: config with hyperparameters
        current_lambda: KL weight (used when sample_weight=False)
        current_lr: current learning rate (= ODE time step Δt)
        fc_layer: nn.Linear layer for momentum buffer access
        optimizer: SGD optimizer with momentum buffers
        memory_bank: LogitMemoryBank for same-class k-NN (optional)
        entropy_ema: EntropyEMA for adaptive H_target bounds (optional)
    Returns:
        total_loss, ce_loss, kl_loss, diagnostics
    """
    B, C = logits.size()
    device = logits.device
    
    # ODE time step = current learning rate (natural Δt)
    eta = current_lr
    
    # Hyperparameters
    tau_min = cfg.la_oracle.get("tau_min", 0.1)
    tau_max = cfg.la_oracle.get("tau_max", 10.0)
    bs_iters = cfg.la_oracle.get("bs_iters", 10)
    
    # Enhancement A: Sample weighting params
    use_sample_weight = cfg.la_oracle.get("use_sample_weight", False)
    w_min = cfg.la_oracle.get("w_min", 0.3)
    w_max = cfg.la_oracle.get("w_max", 3.0)
    ema_decay = cfg.la_oracle.get("ema_decay", 0.99)
    
    # Enhancement B: H* gap amplification (controls "how much to follow future")
    delta_h_alpha = cfg.la_oracle.get("delta_h_alpha", 1.0)
    
    # Enhancement C: Include weight decay in lookahead
    include_wd_in_lookahead = cfg.la_oracle.get("include_wd_in_lookahead", False)
    weight_decay = cfg.train.weight_decay if include_wd_in_lookahead else 0.0
    
    # Enhancement D: Include momentum in lookahead
    include_momentum_in_lookahead = cfg.la_oracle.get("include_momentum_in_lookahead", False)
    momentum = cfg.train.momentum if include_momentum_in_lookahead else 0.0
    
    # Enhancement E: KD-style temperature for gradient smoothing
    # T > 1 makes gradients spread across all classes (reduces FKL tail-ignoring)
    # Separate from τ* which handles entropy matching
    kd_temperature = cfg.la_oracle.get("kd_temperature", 1.0)
    
    # Enhancement F: Option to skip tau_star bisection
    # If false, use fixed tau=1 (simpler, works well when lr is small)
    use_tau_star = cfg.la_oracle.get("use_tau_star", True)
    
    # 1. Current probabilities
    probs = F.softmax(logits, dim=1)
    
    # 2. CE Loss per sample
    ce_per_sample = F.cross_entropy(logits, targets, reduction='none')  # (B,)
    ce_loss = ce_per_sample.mean()
    
    # 3. Oracle computation
    # oracle_type: "lookahead" (기존), "centroid", "knn", "adaptive", "subcluster"
    oracle_type = cfg.la_oracle.get("oracle_type", "lookahead")
    valid_mask = None  # For subcluster mode: marks samples with valid teacher

    with torch.no_grad():
        if oracle_type == "centroid" and memory_bank is not None:
            # Centroid-based oracle: 클래스 평균을 목표로
            blend_alpha = cfg.la_oracle.get("blend_alpha", 0.3)
            z_oracle, centroid_dist = compute_centroid_oracle(
                logits, targets, memory_bank, blend_alpha=blend_alpha
            )
            delta_z = z_oracle - logits  # for compatibility

        elif oracle_type == "knn" and memory_bank is not None:
            # k-NN weighted average oracle
            oracle_k = cfg.la_oracle.get("oracle_k", 5)
            blend_alpha = cfg.la_oracle.get("blend_alpha", 0.3)
            z_oracle, knn_dist = compute_knn_oracle(
                logits, targets, memory_bank, k=oracle_k, blend_alpha=blend_alpha
            )
            delta_z = z_oracle - logits

        elif oracle_type == "adaptive" and memory_bank is not None:
            # Adaptive: confidence에 따라 blend_alpha 조절
            oracle_k = cfg.la_oracle.get("oracle_k", 5)
            z_oracle, difficulty = compute_adaptive_oracle(
                logits, probs, targets, memory_bank, k=oracle_k
            )
            delta_z = z_oracle - logits

        elif oracle_type == "subcluster" and subcluster_bank is not None:
            # SubCluster: logit space에서 가까운 sub-cluster의 centroid
            # k-NN처럼 local하지만 O(k) lookup으로 빠름
            z_oracle, valid_mask = subcluster_bank.get_teacher_logits(logits, targets)
            delta_z = z_oracle - logits
            # valid_mask가 False인 샘플은 teacher가 없음 → CE만 사용

        else:
            # 3a. Lookahead logits (exact, analytical) - compute Δz first
            z_oracle, delta_z = compute_lookahead_logits(
                logits, probs, targets, features, eta=eta, weight_decay=weight_decay,
                fc_layer=fc_layer, optimizer=optimizer, momentum=momentum
            )

        # 3b. Entropy estimation (Shannon or KL)
        entropy_estimator = cfg.la_oracle.get("entropy_estimator", "shannon")

        if entropy_estimator == "kl":
            # Same-class k-NN entropy: logit space k-NN based estimation
            # 같은 클래스 샘플들과의 거리를 기반으로 difficulty 추정
            kl_k = cfg.la_oracle.get("kl_k", 1)
            kl_h_min = cfg.la_oracle.get("kl_h_min", 0.0)
            kl_h_max = cfg.la_oracle.get("kl_h_max", None)  # None = log(C)

            # Combined difficulty: k-NN + confidence
            use_confidence = cfg.la_oracle.get("use_confidence", True)
            conf_weight = cfg.la_oracle.get("conf_weight", 0.5)

            # Lookahead logits + targets로 same-class k-NN 계산
            # Memory bank 사용 시 더 많은 same-class 샘플로 k-NN 계산
            if memory_bank is not None:
                if FAISS_AVAILABLE:
                    # FAISS version (faster)
                    H_raw, epsilon_k = same_class_knn_entropy_faiss(
                        z_oracle, targets, memory_bank, k=kl_k
                    )
                else:
                    # PyTorch version (fallback)
                    H_raw, epsilon_k = same_class_knn_entropy_with_memory(
                        z_oracle, targets, memory_bank, k=kl_k
                    )
            else:
                H_raw, epsilon_k = same_class_knn_entropy(z_oracle, targets, k=kl_k)

            # Handle edge case: all samples have same k-NN distance or NaN
            h_range = H_raw.max() - H_raw.min()

            # Determine H bounds: use EMA if available, else config values
            # When kl_h_min/max are null, use EMA bounds (adaptive) or sensible defaults
            if entropy_ema is not None and entropy_ema.initialized:
                ema_h_min, ema_h_max = entropy_ema.get_bounds()
            else:
                # Fallback defaults when EMA not yet initialized
                ema_h_min = kl_h_min if kl_h_min is not None else 0.5
                ema_h_max = kl_h_max if kl_h_max is not None else math.log(C)

            if torch.isnan(H_raw).any() or torch.isinf(H_raw).any() or h_range < 1e-6:
                # Fallback to uniform H_target in middle of range
                H_target = torch.full_like(H_raw, (ema_h_min + ema_h_max) / 2)
                abs_delta_H = torch.zeros_like(H_raw)
            else:
                # Update H_raw EMA (for tracking, even if not used for normalization)
                if entropy_ema is not None:
                    entropy_ema.update_raw(H_raw)

                if use_confidence:
                    # Combined difficulty: k-NN + confidence
                    # - k-NN 큼 + conf 낮음 → high difficulty → soft teacher
                    # - k-NN 큼 + conf 높음 → medium difficulty → sharp teacher (문제 해결!)
                    difficulty, knn_diff, conf_diff = compute_difficulty_score(
                        H_raw, probs, targets, conf_weight=conf_weight
                    )
                    # Map difficulty [0,1] to H_target [H_min, H_max]
                    H_target = ema_h_min + difficulty * (ema_h_max - ema_h_min)
                    abs_delta_H = difficulty
                else:
                    # 기존 방식: k-NN only
                    # 정방향 매핑: H_raw 큼 (클래스 내 outlier) → H_target 큼
                    H_target = map_kl_to_target_entropy(
                        H_raw, H_min=kl_h_min, H_max=kl_h_max, C=C,
                        entropy_ema=entropy_ema, use_batch_relative=True
                    )
                    abs_delta_H = (H_raw - H_raw.min()) / (h_range + 1e-8)

            H_current = entropy(probs)  # Shannon entropy for diagnostics
        else:
            # Shannon entropy: lookahead-based estimation (기존 방식)
            delta_H_pred, H_current = compute_lookahead_entropy_change(probs, delta_z)

            # Difficulty metric: |ΔH_pred| - how much optimizer wants to move this sample's entropy
            abs_delta_H = torch.abs(delta_H_pred)  # (B,)

            # H* gap amplification (controls "how much to follow future")
            delta_H_goal = delta_h_alpha * delta_H_pred

            # Oracle entropy target
            H_target = H_current + delta_H_goal
            H_target = torch.clamp(H_target, 0.0, math.log(C))  # Valid range
        
        # 3c. Find τ* for oracle entropy matching
        # Modes: "target", "current", "fixed", "none", "zscore"
        tau_star_mode = cfg.la_oracle.get("tau_star_mode", "target")
        tau_fixed_value = cfg.la_oracle.get("tau_fixed_value", 4.0)

        if tau_star_mode == "zscore":
            # Z-score normalization: no temperature needed!
            # Both teacher and student use normalized logits → pure shape matching
            eps = 1e-8
            z_oracle_mean = z_oracle.mean(dim=1, keepdim=True)
            z_oracle_std = z_oracle.std(dim=1, keepdim=True) + eps
            z_oracle_normalized = (z_oracle - z_oracle_mean) / z_oracle_std
            q_star = F.softmax(z_oracle_normalized, dim=1)
            # tau_star is meaningless in zscore mode, set to 1 for logging
            tau_star = torch.ones(B, device=device)
        elif tau_star_mode == "target" and use_tau_star:
            # Match H_target (lookahead entropy)
            tau_star, q_star = find_tau_star(
                z_oracle, H_target,
                tau_min=tau_min, tau_max=tau_max, n_iters=bs_iters
            )
        elif tau_star_mode == "current" and use_tau_star:
            # Match H_current (preserve current entropy, change only shape)
            tau_star, q_star = find_tau_star(
                z_oracle, H_current,
                tau_min=tau_min, tau_max=tau_max, n_iters=bs_iters
            )
        elif tau_star_mode == "fixed":
            # Fixed tau value (e.g., 4.0 like standard KD)
            tau_star = torch.full((B,), tau_fixed_value, device=device)
            q_star = F.softmax(z_oracle / tau_fixed_value, dim=1)
        else:
            # mode="none" or use_tau_star=false: skip bisection, use τ=1
            tau_star = torch.ones(B, device=device)
            q_star = F.softmax(z_oracle, dim=1)
    
    # 6. KL Loss computation
    # F-divergence selection: kl, rkl, js
    divergence_type = cfg.la_oracle.get("divergence_type", "kl")
    T = kd_temperature
    tau_expanded = tau_star.view(-1, 1)  # (B, 1)

    if tau_star_mode == "zscore":
        # Z-score mode: both teacher and student use normalized logits
        # No temperature needed - pure shape matching
        eps = 1e-8
        logits_mean = logits.mean(dim=1, keepdim=True)
        logits_std = logits.std(dim=1, keepdim=True) + eps
        logits_normalized = (logits - logits_mean) / logits_std
        p = F.softmax(logits_normalized, dim=1)
        # q_star already computed with z-score normalization above

        if divergence_type == "kl":
            kl_per_sample = torch.sum(q_star * (torch.log(q_star + 1e-8) - torch.log(p + 1e-8)), dim=1)
        elif divergence_type == "rkl":
            kl_per_sample = torch.sum(p * (torch.log(p + 1e-8) - torch.log(q_star + 1e-8)), dim=1)
        elif divergence_type == "js":
            m = 0.5 * (q_star + p)
            kl_qm = torch.sum(q_star * (torch.log(q_star + 1e-8) - torch.log(m + 1e-8)), dim=1)
            kl_pm = torch.sum(p * (torch.log(p + 1e-8) - torch.log(m + 1e-8)), dim=1)
            kl_per_sample = 0.5 * (kl_qm + kl_pm)
        else:
            raise ValueError(f"Unknown divergence_type: {divergence_type}")

    elif T != 1.0:
        # Apply τ* for shape and T for gradient smoothing
        q_star_T = F.softmax(z_oracle / tau_expanded / T, dim=1)
        p_T = F.softmax(logits / T, dim=1)

        if divergence_type == "kl":
            # Forward KL: KL(q* || p) - mean-seeking
            div_per_sample = torch.sum(q_star_T * (torch.log(q_star_T + 1e-8) - torch.log(p_T + 1e-8)), dim=1)
        elif divergence_type == "rkl":
            # Reverse KL: KL(p || q*) - mode-seeking
            div_per_sample = torch.sum(p_T * (torch.log(p_T + 1e-8) - torch.log(q_star_T + 1e-8)), dim=1)
        elif divergence_type == "js":
            # Jensen-Shannon: symmetric, balanced
            m = 0.5 * (q_star_T + p_T)
            kl_qm = torch.sum(q_star_T * (torch.log(q_star_T + 1e-8) - torch.log(m + 1e-8)), dim=1)
            kl_pm = torch.sum(p_T * (torch.log(p_T + 1e-8) - torch.log(m + 1e-8)), dim=1)
            div_per_sample = 0.5 * (kl_qm + kl_pm)
        else:
            raise ValueError(f"Unknown divergence_type: {divergence_type}")

        # T² scaling to maintain gradient magnitude
        kl_per_sample = T * T * div_per_sample
    else:
        # Standard divergence without additional temperature (T=1)
        p = F.softmax(logits, dim=1)

        if divergence_type == "kl":
            kl_per_sample = torch.sum(q_star * (torch.log(q_star + 1e-8) - torch.log(p + 1e-8)), dim=1)
        elif divergence_type == "rkl":
            kl_per_sample = torch.sum(p * (torch.log(p + 1e-8) - torch.log(q_star + 1e-8)), dim=1)
        elif divergence_type == "js":
            m = 0.5 * (q_star + p)
            kl_qm = torch.sum(q_star * (torch.log(q_star + 1e-8) - torch.log(m + 1e-8)), dim=1)
            kl_pm = torch.sum(p * (torch.log(p + 1e-8) - torch.log(m + 1e-8)), dim=1)
            kl_per_sample = 0.5 * (kl_qm + kl_pm)
        else:
            raise ValueError(f"Unknown divergence_type: {divergence_type}")

    # Apply valid_mask for subcluster mode (mask out KL loss for samples without valid teacher)
    if valid_mask is not None:
        # valid_mask: (B,) bool tensor - True = valid teacher, False = no teacher
        # For invalid samples, KL loss = 0 (only CE loss applies)
        kl_per_sample = kl_per_sample * valid_mask.float()

    # Sample-adaptive λ: selectable method
    # - "loss_magnitude": λ_i = CE_i/(CE_i+KL_i) - auto-balancing
    # - "softmax": λ_i ∝ softmax(|ΔH_pred|) - competition among samples  
    # - "fixed": λ_i = current_lambda - no adaptation
    weight_method = cfg.la_oracle.get("weight_method", "loss_magnitude")
    
    if use_sample_weight and weight_method == "loss_magnitude":
        # Auto-balancing based on loss magnitudes
        ce_detached = ce_per_sample.detach()
        kl_detached = kl_per_sample.detach()
        lambda_i = ce_detached / (ce_detached + kl_detached + 1e-6)
        loss_per_sample = (1 - lambda_i) * ce_per_sample + lambda_i * kl_per_sample
        total_loss = loss_per_sample.mean()
        avg_weight = lambda_i.mean()
        
    elif use_sample_weight and weight_method == "softmax":
        # Competition among samples via softmax
        weight_temp = cfg.la_oracle.get("weight_temperature", 1.0)
        weights = F.softmax(abs_delta_H / (weight_temp + 1e-6), dim=0)
        lambda_i = weights / 2
        loss_per_sample = (1 - lambda_i) * ce_per_sample + lambda_i * kl_per_sample
        total_loss = loss_per_sample.mean()
        avg_weight = lambda_i.mean()
        
    elif use_sample_weight and weight_method == "log_softmax":
        # Less extreme softmax using log to compress range
        # log(|ΔH|) reduces the extreme ratio between samples
        weight_temp = cfg.la_oracle.get("weight_temperature", 1.0)
        log_delta_H = torch.log(abs_delta_H + 1e-6)
        weights = F.softmax(log_delta_H / (weight_temp + 1e-6), dim=0)
        lambda_i = weights / 2
        loss_per_sample = lambda_i * ce_per_sample + (1 - lambda_i) * kl_per_sample
        total_loss = loss_per_sample.mean()
        avg_weight = lambda_i.mean()
        
    elif use_sample_weight and weight_method == "entropy_magnitude":
        # Auto-balancing based on entropy magnitudes
        # λ = H_current / (H_current + H_target)
        # High H(p) → more CE (needs sharpening)
        # Low H(p) → more KL (follow oracle)
        lambda_i = H_current / (H_current + H_target + 1e-6)
        lambda_i /= 2
        loss_per_sample = lambda_i * ce_per_sample + (1 - lambda_i) * kl_per_sample
        total_loss = loss_per_sample.mean()
        avg_weight = lambda_i.mean()
        
    elif use_sample_weight and weight_method == "clamp":
        # Batch-relative difficulty with clamping
        # d_i = |ΔH_pred,i| / batch_mean(|ΔH_pred|)
        # d_i = 1 (average) → λ_i = 0.5
        # d_i ≥ 1 (hard)    → λ_i = 0.5 (clamped max)
        # d_i = 0 (easy)    → λ_i = 0.0
        batch_mean = abs_delta_H.mean() + 1e-6
        d_i = abs_delta_H / batch_mean
        lambda_i = torch.clamp(d_i / 2.0, 0.0, 0.5)
        loss_per_sample = (1 - lambda_i) * ce_per_sample + lambda_i * kl_per_sample
        total_loss = loss_per_sample.mean()
        avg_weight = lambda_i.mean()
        
    else:
        # Fixed global λ (no sample adaptation)
        total_loss = (1 - current_lambda) * ce_loss + current_lambda * kl_per_sample.mean()
        avg_weight = torch.tensor(current_lambda)
    
    # Diagnostics
    with torch.no_grad():
        delta_z_norm = torch.norm(delta_z, dim=1).mean()

    # Update entropy EMA with current batch's Shannon entropy
    if entropy_ema is not None:
        entropy_ema.update(H_current)

    return total_loss, ce_loss, kl_per_sample.mean(), H_current.mean(), H_target.mean(), tau_star.mean(), delta_z_norm, avg_weight, z_oracle


@hydra.main(version_base=None, config_path="distill/conf", config_name="la_oracle")
def main(cfg: DictConfig):
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set seed for reproducibility
    set_seed(42)
    logger.info("Seed set to 42 for reproducibility")
    
    # Data
    logger.info("Preparing Data...")
    
    if cfg.data.get("use_augmentation", True):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        logger.info("Using data augmentation")
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        logger.info("Data augmentation DISABLED")
        
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=cfg.data.num_workers)
    
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers)
    
    # Model
    logger.info("Creating Model (ResNet20)...")
    net = resnet20(num_classes=cfg.model.num_classes).to(device)
    
    optimizer = optim.SGD(net.parameters(), lr=cfg.train.lr, momentum=cfg.train.momentum, weight_decay=cfg.train.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 180, 210], gamma=0.1)
    
    # Memory bank for same-class k-NN (optional)
    use_memory_bank = cfg.la_oracle.get("use_memory_bank", False)
    memory_bank = None
    if use_memory_bank and cfg.la_oracle.get("entropy_estimator", "shannon") == "kl":
        queue_size = cfg.la_oracle.get("memory_queue_size", 64)
        memory_bank = LogitMemoryBank(
            num_classes=cfg.model.num_classes,
            queue_size_per_class=queue_size,
            logit_dim=cfg.model.num_classes,
            device=device
        )
        logger.info(f"Memory bank enabled: {queue_size} samples per class")

    # EntropyEMA for adaptive H_target bounds (optional)
    use_entropy_ema = cfg.la_oracle.get("use_entropy_ema", True)
    entropy_ema = None
    if use_entropy_ema and cfg.la_oracle.get("entropy_estimator", "shannon") == "kl":
        ema_decay = cfg.la_oracle.get("entropy_ema_decay", 0.99)
        # Use sensible defaults when kl_h_min/max are null
        # These will be overwritten by actual observed values on first batch
        initial_h_min = cfg.la_oracle.get("kl_h_min", None)
        initial_h_max = cfg.la_oracle.get("kl_h_max", None)
        if initial_h_min is None:
            initial_h_min = 0.5  # Reasonable starting point (will be updated by EMA)
        if initial_h_max is None:
            initial_h_max = math.log(cfg.model.num_classes)  # log(C)
        entropy_ema = EntropyEMA(
            decay=ema_decay,
            initial_H_min=initial_h_min,
            initial_H_max=initial_h_max
        )
        logger.info(f"EntropyEMA enabled: decay={ema_decay}, initial=[{initial_h_min:.2f}, {initial_h_max:.2f}]")

    # SubClusterBank for fast local centroid oracle (optional)
    oracle_type = cfg.la_oracle.get("oracle_type", "lookahead")
    subcluster_bank = None
    if oracle_type == "subcluster":
        k_clusters = cfg.la_oracle.get("k_clusters", 4)
        subcluster_ema = cfg.la_oracle.get("subcluster_ema_decay", 0.99)
        subcluster_bank = SubClusterBank(
            num_classes=cfg.model.num_classes,
            k_clusters=k_clusters,
            logit_dim=cfg.model.num_classes,
            device=device,
            ema_decay=subcluster_ema
        )
        logger.info(f"SubClusterBank enabled: {k_clusters} clusters/class, EMA decay={subcluster_ema}")

    logger.info("Starting Training...")
    start_time = time.time()

    best_acc = 0.0

    for epoch in range(cfg.train.epochs):
        # Lambda Schedule
        lambda_schedule = cfg.la_oracle.get("lambda_schedule", "cosine")
        lambda_max = cfg.la_oracle.lambda_val
        
        if lambda_schedule == "static":
            current_lambda = lambda_max
        elif lambda_schedule == "cosine":
            progress = epoch / cfg.train.epochs
            current_lambda = lambda_max * (math.sin(math.pi / 2 * progress) ** 2)
        else:
            current_lambda = lambda_max
        
        net.train()
        train_loss = 0
        train_ce = 0
        train_kl = 0
        correct = 0
        total = 0
        
        avg_h_cur = 0
        avg_h_tgt = 0
        avg_tau = 0
        avg_w = 0
        num_batches = 0
        
        pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{cfg.train.epochs}")
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward with features for analytical lookahead
            logits, features = net.forward_with_features(inputs)
            
            # Get current learning rate from optimizer (= ODE time step Δt)
            current_lr = optimizer.param_groups[0]['lr']
            
            loss, ce, kl, h_cur, h_tgt, tau, dz_norm, w_avg, z_oracle = la_oracle_loss(
                logits, features, targets, cfg, current_lambda, current_lr,
                fc_layer=net.fc, optimizer=optimizer, memory_bank=memory_bank,
                entropy_ema=entropy_ema, subcluster_bank=subcluster_bank
            )

            # Update memory bank BEFORE optimizer step (same model state as k-NN query)
            # This ensures memory bank samples and current batch are from same model
            if memory_bank is not None:
                memory_bank.update(z_oracle, targets)

            # Update SubClusterBank with current logits
            if subcluster_bank is not None:
                subcluster_bank.update(logits.detach(), targets)

            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_ce += ce.item()
            train_kl += kl.item()
            
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            avg_h_cur += h_cur.item()
            avg_h_tgt += h_tgt.item()
            avg_tau += tau.item()
            avg_w += w_avg.item() if isinstance(w_avg, torch.Tensor) else w_avg
            num_batches += 1
            
            pbar.set_postfix({
                'Loss': f"{train_loss/num_batches:.4f}",
                'Acc': f"{100.*correct/total:.2f}%",
                'H': f"{avg_h_cur/num_batches:.2f}",
                'H*': f"{avg_h_tgt/num_batches:.2f}",
                'τ': f"{avg_tau/num_batches:.2f}",
                'w': f"{avg_w/num_batches:.2f}",
                'λ': f"{current_lambda:.3f}"
            })
            
            if cfg.la_oracle.get("dry_run", False):
                logger.info(f"Dry Run | Loss: {loss.item():.4f} | CE: {ce.item():.4f} | "
                           f"KL: {kl.item():.4f} | H: {h_cur.item():.2f} | H*: {h_tgt.item():.2f} | "
                           f"τ*: {tau.item():.2f} | w: {w_avg:.2f}")
                logger.info("Dry run complete.")
                return

        scheduler.step()
        
        # Validation
        net.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = F.cross_entropy(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
        
        test_acc = 100. * test_correct / test_total
        train_acc = 100. * correct / total
        
        if test_acc > best_acc:
            best_acc = test_acc
        
        logger.info(f"Epoch {epoch+1}/{cfg.train.epochs} | "
                    f"Train: {train_acc:.2f}% | Test: {test_acc:.2f}% | Best: {best_acc:.2f}% | "
                    f"H: {avg_h_cur/num_batches:.2f} | H*: {avg_h_tgt/num_batches:.2f} | "
                    f"τ: {avg_tau/num_batches:.2f} | λ: {current_lambda:.3f}")

    total_time = time.time() - start_time
    logger.info(f"Training Finished. Best Acc: {best_acc:.2f}% | Time: {total_time/60:.2f} mins")


if __name__ == '__main__':
    main()
