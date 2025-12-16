"""
LA-Oracle 시각화 스크립트
========================
학습 중간에 현재 모델의 상태를 직관적으로 이해하기 위한 시각화.

시각화 내용:
1. 샘플별 Student vs Oracle 분포 비교
2. k-NN distance와 H_target의 관계
3. τ* (temperature)의 분포
4. Logit space에서의 클래스 분포 (t-SNE)
5. KL divergence 분포
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.manifold import TSNE
import math
import os

# Import from train_la_oracle
from models import resnet20

# FAISS for fast k-NN
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# CIFAR-100 class names (일부만)
CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
    'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
    'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]


def entropy(probs):
    """Compute entropy of probability distributions."""
    return -torch.sum(probs * torch.log(probs + 1e-8), dim=1)


def compute_ce_gradient(probs, targets):
    """Compute CE gradient w.r.t. logits: g = p - e_y"""
    B, C = probs.size()
    one_hot = torch.zeros_like(probs)
    one_hot.scatter_(1, targets.view(-1, 1), 1.0)
    return probs - one_hot


def compute_lookahead_logits(logits, probs, targets, features, eta, weight_decay=0.0):
    """Compute lookahead oracle logits."""
    g_CE = compute_ce_gradient(probs, targets)
    h_norm_sq = torch.sum(features ** 2, dim=1, keepdim=True)
    scale = h_norm_sq + 1.0
    ce_term = g_CE * scale
    wd_term = weight_decay * logits
    delta_z = -eta * (ce_term + wd_term)
    z_oracle = logits + delta_z
    return z_oracle, delta_z


def find_tau_star(z_oracle, H_target, tau_min=0.1, tau_max=3.0, bs_iters=10):
    """Find τ* via bisection such that H(softmax(z_oracle/τ*)) = H_target."""
    B = z_oracle.size(0)
    tau_lo = torch.full((B,), tau_min, device=z_oracle.device)
    tau_hi = torch.full((B,), tau_max, device=z_oracle.device)

    for _ in range(bs_iters):
        tau = (tau_lo + tau_hi) / 2.0
        tau_expanded = tau.view(-1, 1)
        q = F.softmax(z_oracle / tau_expanded, dim=1)
        H_q = entropy(q)

        tau_lo = torch.where(H_q < H_target, tau, tau_lo)
        tau_hi = torch.where(H_q >= H_target, tau, tau_hi)

    tau_star = (tau_lo + tau_hi) / 2.0
    q_star = F.softmax(z_oracle / tau_star.view(-1, 1), dim=1)
    return tau_star, q_star


class LogitMemoryBank:
    """Per-class FIFO queue for storing z_oracle logits."""
    def __init__(self, num_classes, queue_size, feature_dim, device):
        self.num_classes = num_classes
        self.queue_size = queue_size
        self.feature_dim = feature_dim
        self.device = device
        self.queues = {c: [] for c in range(num_classes)}

    @torch.no_grad()
    def update(self, z_oracle, targets):
        z_oracle = z_oracle.detach()
        for i, target in enumerate(targets):
            c = target.item()
            self.queues[c].append(z_oracle[i].cpu())
            if len(self.queues[c]) > self.queue_size:
                self.queues[c].pop(0)

    def get_class_samples(self, class_idx):
        if len(self.queues[class_idx]) == 0:
            return torch.empty(0, self.feature_dim, device=self.device)
        return torch.stack(self.queues[class_idx]).to(self.device)


def same_class_knn_distances(z, targets, memory_bank, k=3):
    """Compute k-NN distances within same class using memory bank."""
    B, d = z.size()
    device = z.device

    epsilon_k = torch.full((B,), float('inf'), device=device)
    unique_classes = targets.unique()

    for class_idx in unique_classes:
        class_idx_item = class_idx.item()
        batch_mask = (targets == class_idx)
        batch_indices = batch_mask.nonzero(as_tuple=True)[0]
        z_class = z[batch_mask]
        n_batch = z_class.size(0)

        if n_batch == 0:
            continue

        mem_samples = memory_bank.get_class_samples(class_idx_item)
        n_mem = mem_samples.size(0)

        if n_mem >= k:
            dist_matrix = torch.cdist(z_class, mem_samples, p=2)
            kth_dists, _ = torch.kthvalue(dist_matrix, k, dim=1)
            epsilon_k[batch_indices] = kth_dists
        elif n_mem > 0:
            combined = torch.cat([mem_samples, z_class], dim=0)
            dist_matrix = torch.cdist(z_class, combined, p=2)
            dist_matrix[:, n_mem:n_mem+n_batch] = float('inf')
            for i in range(n_batch):
                dist_matrix[i, n_mem + i] = float('inf')

            if combined.size(0) > k:
                kth_dists, _ = torch.kthvalue(dist_matrix, min(k, combined.size(0)), dim=1)
                epsilon_k[batch_indices] = kth_dists

    epsilon_k = torch.clamp(epsilon_k, min=1e-10)
    return epsilon_k


def map_knn_to_h_target(epsilon_k, h_min, h_max, d=100):
    """Map k-NN distances to H_target."""
    H_raw = d * torch.log(epsilon_k)
    H_raw_min = H_raw.min()
    H_raw_max = H_raw.max()
    h_range = H_raw_max - H_raw_min

    if h_range < 1e-6:
        H_normalized = torch.full_like(H_raw, 0.5)
    else:
        H_normalized = (H_raw - H_raw_min) / (h_range + 1e-8)

    H_target = h_min + H_normalized * (h_max - h_min)
    return H_target, H_raw


def visualize_la_oracle(model, dataloader, memory_bank, device,
                        save_path='la_oracle_visualization.png',
                        num_samples=8, k=3, eta=0.05, weight_decay=5e-4):
    """
    LA-Oracle 시각화

    Args:
        model: trained or partially trained model
        dataloader: data loader
        memory_bank: LogitMemoryBank instance (with some samples)
        device: cuda/cpu
        save_path: where to save the figure
        num_samples: number of samples to visualize in detail
        k: k for k-NN
        eta: learning rate (for lookahead)
        weight_decay: weight decay
    """
    model.eval()

    # Get a batch
    images, targets = next(iter(dataloader))
    images, targets = images.to(device), targets.to(device)
    B = images.size(0)

    with torch.no_grad():
        # Forward pass
        logits, features = model.forward_with_features(images)
        probs = F.softmax(logits, dim=1)
        H_current = entropy(probs)

        # Compute lookahead logits
        z_oracle, delta_z = compute_lookahead_logits(
            logits, probs, targets, features, eta, weight_decay
        )

        # Compute k-NN distances
        epsilon_k = same_class_knn_distances(z_oracle, targets, memory_bank, k=k)

        # Compute H_target
        h_min, h_max = H_current.min().item(), H_current.max().item()
        H_target, H_raw = map_knn_to_h_target(epsilon_k, h_min, h_max)

        # Find tau* and oracle distribution
        tau_star, q_star = find_tau_star(z_oracle, H_target)
        H_oracle = entropy(q_star)

        # Compute KL divergence per sample
        kl_per_sample = torch.sum(q_star * (torch.log(q_star + 1e-8) - torch.log(probs + 1e-8)), dim=1)

    # Convert to numpy
    images_np = images.cpu().numpy()
    targets_np = targets.cpu().numpy()
    probs_np = probs.cpu().numpy()
    q_star_np = q_star.cpu().numpy()
    logits_np = logits.cpu().numpy()
    z_oracle_np = z_oracle.cpu().numpy()
    H_current_np = H_current.cpu().numpy()
    H_target_np = H_target.cpu().numpy()
    H_oracle_np = H_oracle.cpu().numpy()
    tau_star_np = tau_star.cpu().numpy()
    epsilon_k_np = epsilon_k.cpu().numpy()
    H_raw_np = H_raw.cpu().numpy()
    kl_np = kl_per_sample.cpu().numpy()

    # Create figure
    fig = plt.figure(figsize=(28, 20))
    gs = gridspec.GridSpec(4, 5, figure=fig, hspace=0.35, wspace=0.3)

    # =========================================================================
    # Row 1: Sample visualizations (image + student vs oracle distribution)
    # =========================================================================
    # Select samples with varying difficulty (by k-NN distance)
    sorted_indices = np.argsort(epsilon_k_np)
    sample_indices = [
        sorted_indices[0],  # easiest (closest k-NN)
        sorted_indices[B//4],
        sorted_indices[B//2],
        sorted_indices[3*B//4],
        sorted_indices[-1],  # hardest (farthest k-NN)
    ]
    sample_indices = sample_indices[:min(5, num_samples)]

    for plot_idx, sample_idx in enumerate(sample_indices):
        ax = fig.add_subplot(gs[0, plot_idx])

        # Show image
        img = images_np[sample_idx].transpose(1, 2, 0)
        img = (img * np.array([0.2023, 0.1994, 0.2010])) + np.array([0.4914, 0.4822, 0.4465])
        img = np.clip(img, 0, 1)
        ax.imshow(img)

        true_class = targets_np[sample_idx]
        pred_class = np.argmax(probs_np[sample_idx])

        title = f'Class: {CIFAR100_CLASSES[true_class][:8]}\n'
        title += f'k-NN dist: {epsilon_k_np[sample_idx]:.2f}\n'
        title += f'τ*: {tau_star_np[sample_idx]:.2f}, KL: {kl_np[sample_idx]:.3f}'
        ax.set_title(title, fontsize=9)
        ax.axis('off')

    # =========================================================================
    # Row 2: Distribution comparison for selected samples
    # =========================================================================
    for plot_idx, sample_idx in enumerate(sample_indices):
        ax = fig.add_subplot(gs[1, plot_idx])

        true_class = targets_np[sample_idx]

        # Get top-10 classes by oracle probability
        top_k = 10
        top_indices = np.argsort(q_star_np[sample_idx])[-top_k:][::-1]

        x = np.arange(top_k)
        width = 0.35

        student_probs = probs_np[sample_idx][top_indices]
        oracle_probs = q_star_np[sample_idx][top_indices]

        bars1 = ax.bar(x - width/2, student_probs, width, label='Student p(x)', alpha=0.8, color='steelblue')
        bars2 = ax.bar(x + width/2, oracle_probs, width, label='Oracle q*(x)', alpha=0.8, color='coral')

        # Highlight true class
        for i, idx in enumerate(top_indices):
            if idx == true_class:
                ax.axvline(i, color='green', linestyle='--', alpha=0.5, linewidth=2)
                ax.annotate('GT', (i, max(student_probs[i], oracle_probs[i]) + 0.02),
                           ha='center', fontsize=8, color='green')

        ax.set_xticks(x)
        ax.set_xticklabels([CIFAR100_CLASSES[i][:5] for i in top_indices], rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('Probability', fontsize=8)
        ax.set_title(f'H_cur={H_current_np[sample_idx]:.2f} → H_tgt={H_target_np[sample_idx]:.2f}', fontsize=9)
        if plot_idx == 0:
            ax.legend(fontsize=7, loc='upper right')

    # =========================================================================
    # Row 3: Statistics plots
    # =========================================================================

    # 3-1: k-NN distance vs H_target
    ax1 = fig.add_subplot(gs[2, 0])
    scatter = ax1.scatter(epsilon_k_np, H_target_np, c=kl_np, cmap='viridis', alpha=0.7, s=20)
    ax1.set_xlabel('k-NN Distance (ε_k)', fontsize=10)
    ax1.set_ylabel('H_target', fontsize=10)
    ax1.set_title('k-NN Distance → H_target Mapping', fontsize=11)
    plt.colorbar(scatter, ax=ax1, label='KL(q*||p)')

    # Add trend line
    z = np.polyfit(epsilon_k_np, H_target_np, 1)
    p = np.poly1d(z)
    x_line = np.linspace(epsilon_k_np.min(), epsilon_k_np.max(), 100)
    ax1.plot(x_line, p(x_line), 'r--', alpha=0.8, label=f'Linear fit')
    ax1.legend(fontsize=8)

    # 3-2: H_current vs H_target
    ax2 = fig.add_subplot(gs[2, 1])
    scatter2 = ax2.scatter(H_current_np, H_target_np, c=tau_star_np, cmap='coolwarm', alpha=0.7, s=20)
    ax2.plot([h_min, h_max], [h_min, h_max], 'k--', alpha=0.5, label='H_cur = H_tgt')
    ax2.set_xlabel('H_current (Student)', fontsize=10)
    ax2.set_ylabel('H_target (Oracle)', fontsize=10)
    ax2.set_title('Current vs Target Entropy', fontsize=11)
    plt.colorbar(scatter2, ax=ax2, label='τ*')
    ax2.legend(fontsize=8)

    # 3-3: τ* distribution
    ax3 = fig.add_subplot(gs[2, 2])
    ax3.hist(tau_star_np, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax3.axvline(1.0, color='red', linestyle='--', label='τ=1 (no scaling)')
    ax3.axvline(tau_star_np.mean(), color='green', linestyle='-', label=f'Mean={tau_star_np.mean():.2f}')
    ax3.set_xlabel('τ* (Temperature)', fontsize=10)
    ax3.set_ylabel('Count', fontsize=10)
    ax3.set_title('τ* Distribution', fontsize=11)
    ax3.legend(fontsize=8)

    # 3-4: KL divergence distribution
    ax4 = fig.add_subplot(gs[2, 3])
    ax4.hist(kl_np, bins=30, alpha=0.7, color='coral', edgecolor='black')
    ax4.axvline(kl_np.mean(), color='green', linestyle='-', label=f'Mean={kl_np.mean():.3f}')
    ax4.set_xlabel('KL(q* || p)', fontsize=10)
    ax4.set_ylabel('Count', fontsize=10)
    ax4.set_title('KL Divergence Distribution', fontsize=11)
    ax4.legend(fontsize=8)

    # =========================================================================
    # Row 4: t-SNE visualization of logit space
    # =========================================================================

    # 4-1: t-SNE of z_oracle colored by class
    ax5 = fig.add_subplot(gs[3, 0:3])

    # Use subset for t-SNE (faster)
    n_tsne = min(B, 200)
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n_tsne-1))
    z_oracle_2d = tsne.fit_transform(z_oracle_np[:n_tsne])

    # Color by class (use first 10 classes for visibility)
    unique_classes = np.unique(targets_np[:n_tsne])
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_classes)))

    for i, c in enumerate(unique_classes[:15]):  # Show first 15 classes
        mask = targets_np[:n_tsne] == c
        if mask.sum() > 0:
            ax5.scatter(z_oracle_2d[mask, 0], z_oracle_2d[mask, 1],
                       c=[colors[i % len(colors)]], label=CIFAR100_CLASSES[c][:6],
                       alpha=0.7, s=30)

    ax5.set_xlabel('t-SNE 1', fontsize=10)
    ax5.set_ylabel('t-SNE 2', fontsize=10)
    ax5.set_title('z_oracle in Logit Space (t-SNE, colored by class)', fontsize=11)
    ax5.legend(fontsize=7, loc='upper right', ncol=3, bbox_to_anchor=(1.0, 1.0))

    # 4-2: t-SNE colored by k-NN distance (difficulty)
    ax6 = fig.add_subplot(gs[3, 3:5])
    scatter6 = ax6.scatter(z_oracle_2d[:, 0], z_oracle_2d[:, 1],
                           c=epsilon_k_np[:n_tsne], cmap='RdYlGn_r', alpha=0.7, s=30)
    ax6.set_xlabel('t-SNE 1', fontsize=10)
    ax6.set_ylabel('t-SNE 2', fontsize=10)
    ax6.set_title('z_oracle colored by k-NN Distance (Red=Hard, Green=Easy)', fontsize=11)
    plt.colorbar(scatter6, ax=ax6, label='k-NN Distance')

    # =========================================================================
    # Add summary text
    # =========================================================================
    summary_text = (
        f"Summary Statistics:\n"
        f"─────────────────────\n"
        f"Batch size: {B}\n"
        f"k for k-NN: {k}\n"
        f"η (lr): {eta}\n"
        f"─────────────────────\n"
        f"H_current: {H_current_np.mean():.3f} ± {H_current_np.std():.3f}\n"
        f"H_target:  {H_target_np.mean():.3f} ± {H_target_np.std():.3f}\n"
        f"H_oracle:  {H_oracle_np.mean():.3f} ± {H_oracle_np.std():.3f}\n"
        f"─────────────────────\n"
        f"τ*: {tau_star_np.mean():.3f} ± {tau_star_np.std():.3f}\n"
        f"KL: {kl_np.mean():.4f} ± {kl_np.std():.4f}\n"
        f"ε_k: {epsilon_k_np.mean():.2f} ± {epsilon_k_np.std():.2f}"
    )

    fig.text(0.98, 0.98, summary_text, transform=fig.transFigure, fontsize=9,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             family='monospace')

    plt.suptitle('LA-Oracle: Same-Class k-NN Entropy Estimation Visualization',
                 fontsize=14, fontweight='bold', y=0.995)

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Visualization saved to: {save_path}")
    return save_path


def main():
    """Load a checkpoint and visualize."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test
    )
    testloader = DataLoader(testset, batch_size=64, shuffle=True, num_workers=2)

    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_test
    )
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

    # Model
    model = resnet20(num_classes=100).to(device)

    # Try to load checkpoint
    checkpoint_path = 'best_student.pt'
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print("No checkpoint found, using random initialization")

    # Build memory bank by running through some training data
    print("Building memory bank...")
    memory_bank = LogitMemoryBank(
        num_classes=100,
        queue_size=128,
        feature_dim=100,  # logit dim
        device=device
    )

    model.eval()
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(trainloader):
            if batch_idx >= 50:  # Use 50 batches to fill memory bank
                break
            images, targets = images.to(device), targets.to(device)
            logits, features = model.forward_with_features(images)
            probs = F.softmax(logits, dim=1)
            z_oracle, _ = compute_lookahead_logits(
                logits, probs, targets, features, eta=0.05, weight_decay=5e-4
            )
            memory_bank.update(z_oracle, targets)

    print(f"Memory bank built with ~{50*64} samples")

    # Visualize
    visualize_la_oracle(
        model=model,
        dataloader=testloader,
        memory_bank=memory_bank,
        device=device,
        save_path='la_oracle_visualization.png',
        num_samples=5,
        k=3,
        eta=0.05,
        weight_decay=5e-4
    )


if __name__ == '__main__':
    main()
