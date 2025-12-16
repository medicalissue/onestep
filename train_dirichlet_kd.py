"""
Dirichlet Knowledge Distillation
================================

Core Idea:
  Teacher(ResNet56) -> Student(ResNet20) KD on CIFAR-100

  Unlike standard KD which only transfers class probabilities,
  Dirichlet KD transfers BOTH:
    1. Class relations (shape) via softmax(z/T)
    2. Teacher confidence (scale) via evidence

  Loss:
    L = (1-λ)·CE(y, softmax(z_S)) + λ·KL(Dir(α_T) || Dir(α_S))

  Evidence Types:
    - "norm": α = softmax(z/T) * κ * ||z||
              Uses L2 norm as confidence measure
    - "evidential": α = κ * softplus(z) + 1
              Evidential Deep Learning style

Hyperparameters:
  T: shape temperature (controls class relation smoothness)
  κ: evidence scale (how strongly to interpret logit magnitude as confidence)
  λ: distillation weight
  evidence_type: "norm" or "evidential"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import logging
import sys
import random
import numpy as np
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

from models import resnet20, resnet56

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def logits_to_dirichlet_params(logits, T, kappa, evidence_type="norm", alpha_min=0.01):
    """
    Convert logits to Dirichlet parameters.

    Evidence Types:
      - "norm": α = softmax(z/T) * κ * ||z||
               Uses L2 norm of logits as confidence/evidence
               α_0 ≈ κ * ||z|| (total evidence)

      - "evidential": α = κ * softplus(z) + 1
               Evidential Deep Learning style
               Each logit directly maps to evidence
               +1 is the uniform prior

    Args:
        logits: (B, K) raw logits
        T: temperature for shape (only used for "norm")
        kappa: scale factor for evidence
        evidence_type: "norm" or "evidential"
        alpha_min: minimum value for each α_i (for numerical stability)

    Returns:
        alpha: (B, K) Dirichlet parameters
        pi: (B, K) shape (softmax probabilities or normalized alpha)
        scale: (B,) evidence scale (α_0 = sum of alphas)
    """
    B, K = logits.size()

    if evidence_type == "norm":
        # Norm-based: α = softmax(z/T) * κ * ||z||
        # Shape from temperature-scaled softmax
        pi = F.softmax(logits / T, dim=1)  # (B, K)

        # Scale from L2 norm of logits
        z_norm = torch.norm(logits, dim=1, keepdim=True)  # (B, 1)
        scale_per_sample = kappa * z_norm  # (B, 1)

        # Ensure minimum scale
        scale_per_sample = torch.clamp(scale_per_sample, min=alpha_min * K)

        # α = scale * π
        alpha = scale_per_sample * pi  # (B, K)
        alpha = torch.clamp(alpha, min=alpha_min)

        scale = alpha.sum(dim=1)  # (B,) total evidence

    elif evidence_type == "evidential":
        # Evidential: α = κ * softplus(z) + 1
        # Each logit directly becomes evidence
        evidence = F.softplus(logits)  # (B, K), all positive
        alpha = kappa * evidence + 1  # +1 is uniform prior
        alpha = torch.clamp(alpha, min=alpha_min)

        # For consistency, compute pi as normalized alpha
        scale = alpha.sum(dim=1)  # (B,) α_0
        pi = alpha / scale.unsqueeze(1)  # (B, K) E[Dir(α)]

    else:
        raise ValueError(f"Unknown evidence_type: {evidence_type}")

    return alpha, pi, scale


def dirichlet_kl_single(alpha_p, alpha_q):
    """
    KL(Dir(α_p) || Dir(α_q)) - single direction KL divergence.

    Args:
        alpha_p: (B, K) Dirichlet params of P
        alpha_q: (B, K) Dirichlet params of Q

    Returns:
        kl: (B,) KL divergence per sample
    """
    alpha_p0 = alpha_p.sum(dim=1)  # (B,)
    alpha_q0 = alpha_q.sum(dim=1)  # (B,)

    term1 = torch.lgamma(alpha_p0) - torch.lgamma(alpha_q0)
    term2 = (torch.lgamma(alpha_q) - torch.lgamma(alpha_p)).sum(dim=1)

    psi_alpha_p = torch.digamma(alpha_p)
    psi_alpha_p0 = torch.digamma(alpha_p0).unsqueeze(1)
    term3 = ((alpha_p - alpha_q) * (psi_alpha_p - psi_alpha_p0)).sum(dim=1)

    return term1 + term2 + term3


def dirichlet_divergence(alpha_t, alpha_s, mode="kl", scale_by_alpha0="none"):
    """
    Divergence between two Dirichlet distributions.

    Args:
        alpha_t: (B, K) Teacher Dirichlet params
        alpha_s: (B, K) Student Dirichlet params
        mode: "kl" (KL(T||S)), "rkl" (KL(S||T)), "js" (Jensen-Shannon)
        scale_by_alpha0: T² analogue for gradient restoration
            - "none" or False: no scaling
            - "grad": α₀ * KL, gradient flows through α₀ (penalizes high confidence)
            - "detach": α₀.detach() * KL, only magnitude correction (recommended)

    Returns:
        div: (B,) divergence per sample

    Note on scale_by_alpha0:
        Hinton KD: gradient ∝ 1/T² → multiply by T² to restore
        Dirichlet KD: gradient ∝ 1/α₀ → multiply by α₀ to restore

        Problem with "grad": ∂L/∂α includes KL term → penalizes high α₀
        Solution "detach": only corrects gradient magnitude, doesn't penalize confidence
    """
    if mode == "kl":
        # Forward KL: KL(T || S) - standard KD direction
        div = dirichlet_kl_single(alpha_t, alpha_s)
    elif mode == "rkl":
        # Reverse KL: KL(S || T)
        div = dirichlet_kl_single(alpha_s, alpha_t)
    elif mode == "js":
        # Jensen-Shannon: 0.5 * KL(T||M) + 0.5 * KL(S||M) where M = (T+S)/2
        alpha_m = (alpha_t + alpha_s) / 2
        kl_tm = dirichlet_kl_single(alpha_t, alpha_m)
        kl_sm = dirichlet_kl_single(alpha_s, alpha_m)
        div = 0.5 * kl_tm + 0.5 * kl_sm
    else:
        raise ValueError(f"Unknown divergence mode: {mode}")

    # T² analogue: scale by student concentration to restore gradient magnitude
    if scale_by_alpha0 == "grad" or scale_by_alpha0 is True:
        # Gradient flows through α₀ - penalizes high confidence
        alpha_s0 = alpha_s.sum(dim=1)  # (B,)
        div = alpha_s0 * div
    elif scale_by_alpha0 == "detach":
        # Detached - only magnitude correction, doesn't penalize confidence
        alpha_s0 = alpha_s.sum(dim=1).detach()  # (B,)
        div = alpha_s0 * div

    return div


def train_epoch(student, teacher, train_loader, optimizer, device, cfg):
    """Train one epoch with Dirichlet KD."""
    student.train()
    teacher.eval()

    total_loss = 0
    total_ce = 0
    total_dir_kl = 0
    total_scale_t = 0
    total_scale_s = 0
    correct = 0
    total = 0

    T = cfg.dirichlet.T
    kappa = cfg.dirichlet.kappa
    lambda_kd = cfg.dirichlet.lambda_kd
    evidence_type = cfg.dirichlet.get("evidence_type", "norm")

    pbar = tqdm(train_loader, desc="Training")

    for images, targets in pbar:
        images, targets = images.to(device), targets.to(device)

        # Forward
        with torch.no_grad():
            teacher_logits = teacher(images)
        student_logits = student(images)

        # Convert to Dirichlet parameters
        alpha_t, pi_t, scale_t = logits_to_dirichlet_params(
            teacher_logits, T, kappa, evidence_type=evidence_type
        )
        alpha_s, pi_s, scale_s = logits_to_dirichlet_params(
            student_logits, T, kappa, evidence_type=evidence_type
        )

        # CE loss (standard classification)
        ce_loss = F.cross_entropy(student_logits, targets)

        # Dirichlet divergence loss
        divergence_mode = cfg.dirichlet.get("divergence", "kl")
        scale_by_alpha0 = cfg.dirichlet.get("scale_by_alpha0", False)
        dir_div = dirichlet_divergence(
            alpha_t, alpha_s, mode=divergence_mode, scale_by_alpha0=scale_by_alpha0
        )  # (B,)
        dir_kl_loss = dir_div.mean()

        # Optionally normalize by K for class-count invariance
        # Options: "none", "k", "sqrt_k"
        normalize_by_k = cfg.dirichlet.get("normalize_by_k", "none")
        if normalize_by_k == "k" or normalize_by_k is True:
            K = student_logits.size(1)
            dir_kl_loss = dir_kl_loss / K
        elif normalize_by_k == "sqrt_k":
            K = student_logits.size(1)
            dir_kl_loss = dir_kl_loss / (K ** 0.5)

        # Combined loss
        loss = (1 - lambda_kd) * ce_loss + lambda_kd * dir_kl_loss

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Stats
        total_loss += loss.item()
        total_ce += ce_loss.item()
        total_dir_kl += dir_kl_loss.item()
        total_scale_t += scale_t.mean().item()
        total_scale_s += scale_s.mean().item()
        _, predicted = student_logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'CE': f'{ce_loss.item():.3f}',
            'DirKL': f'{dir_kl_loss.item():.3f}',
            's_T': f'{scale_t.mean().item():.1f}',
            's_S': f'{scale_s.mean().item():.1f}',
            'acc': f'{100.*correct/total:.1f}%'
        })

    n_batches = len(train_loader)
    return {
        'loss': total_loss / n_batches,
        'ce': total_ce / n_batches,
        'dir_kl': total_dir_kl / n_batches,
        'scale_t': total_scale_t / n_batches,
        'scale_s': total_scale_s / n_batches,
        'acc': 100. * correct / total
    }


@torch.no_grad()
def evaluate(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()
    correct = 0
    total = 0

    for images, targets in test_loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return 100. * correct / total


@hydra.main(config_path="distill/conf", config_name="dirichlet_kd", version_base=None)
def main(cfg: DictConfig):
    logger.info("=" * 60)
    logger.info("Dirichlet Knowledge Distillation")
    logger.info("=" * 60)
    logger.info(f"\n{OmegaConf.to_yaml(cfg)}")

    # Set seed
    set_seed(cfg.train.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Data transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    # Datasets
    train_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.data.batch_size,
        shuffle=True, num_workers=cfg.data.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.data.batch_size,
        shuffle=False, num_workers=cfg.data.num_workers, pin_memory=True
    )

    # =========================================================================
    # Load teacher model
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Loading teacher model (ResNet56)")
    logger.info("=" * 60)

    teacher = resnet56(num_classes=cfg.model.num_classes).to(device)
    teacher_ckpt = torch.load(cfg.kd.teacher_checkpoint, map_location=device)
    teacher.load_state_dict(teacher_ckpt)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    logger.info(f"Loaded teacher from {cfg.kd.teacher_checkpoint}")

    teacher_acc = evaluate(teacher, test_loader, device)
    logger.info(f"Teacher test accuracy: {teacher_acc:.2f}%")

    # =========================================================================
    # Train student with Dirichlet KD
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Training student with Dirichlet KD")
    logger.info(f"  evidence_type = {cfg.dirichlet.get('evidence_type', 'norm')}")
    logger.info(f"  T (shape temp) = {cfg.dirichlet.T}")
    logger.info(f"  κ (evidence scale) = {cfg.dirichlet.kappa}")
    logger.info(f"  λ (KD weight) = {cfg.dirichlet.lambda_kd}")
    logger.info("=" * 60)

    # Initialize fresh student
    student = resnet20(num_classes=cfg.model.num_classes).to(device)

    optimizer = optim.SGD(
        student.parameters(),
        lr=cfg.train.lr,
        momentum=cfg.train.momentum,
        weight_decay=cfg.train.weight_decay
    )

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg.train.milestones,
        gamma=cfg.train.gamma
    )

    best_acc = 0.0

    for epoch in range(cfg.train.epochs):
        logger.info(f"\nEpoch {epoch+1}/{cfg.train.epochs} (lr={optimizer.param_groups[0]['lr']:.4f})")

        # Train
        train_stats = train_epoch(student, teacher, train_loader, optimizer, device, cfg)

        # Evaluate
        test_acc = evaluate(student, test_loader, device)

        # Update best
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(student.state_dict(), 'best_dirichlet_kd_student.pt')

        logger.info(
            f"  Train: loss={train_stats['loss']:.3f}, CE={train_stats['ce']:.3f}, "
            f"DirKL={train_stats['dir_kl']:.3f}, acc={train_stats['acc']:.1f}%"
        )
        logger.info(
            f"  Test: acc={test_acc:.2f}% (best={best_acc:.2f}%)"
        )
        logger.info(
            f"  Scale: teacher={train_stats['scale_t']:.1f}, student={train_stats['scale_s']:.1f}"
        )

        scheduler.step()

    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info(f"Best test accuracy: {best_acc:.2f}%")
    logger.info(f"Teacher accuracy: {teacher_acc:.2f}%")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
