"""
Entropy-Matched Knowledge Distillation (Sample-wise)
=====================================================

Core Idea:
  Teacher(ResNet56) -> Student(ResNet20) KD on CIFAR-100

  Key difference from standard KD:
  - Sample-wise entropy matching: H(teacher_i / τ_i) = H(student_ref_i)
  - τ_i found via Safe Newton method (on-the-fly per batch)
  - No T^2 scaling needed

  Since entropy is matched per sample, gradient scales are naturally aligned.
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


def entropy(probs):
    """Compute entropy of probability distributions. probs: (B, C) -> (B,)"""
    return -torch.sum(probs * torch.log(probs + 1e-8), dim=1)


def entropy_with_tau(logits, tau):
    """
    Compute entropy with temperature τ.

    Args:
        logits: (B, C)
        tau: (B,) or scalar
    Returns:
        H: (B,)
    """
    if isinstance(tau, (int, float)):
        scaled_logits = logits / tau
    else:
        scaled_logits = logits / tau.unsqueeze(1)
    probs = F.softmax(scaled_logits, dim=1)
    return entropy(probs)


def entropy_derivative_wrt_tau(logits, tau):
    """
    Compute dH/dτ analytically.

    dH/dτ = (1/τ²) * [E_p[z·log p] + H·E_p[z]]

    Args:
        logits: (B, C) - z
        tau: (B,) - temperature per sample
    Returns:
        dH_dtau: (B,)
        H: (B,) - current entropy
    """
    # Compute p = softmax(z/τ)
    scaled_logits = logits / tau.unsqueeze(1)  # (B, C)
    probs = F.softmax(scaled_logits, dim=1)  # (B, C)
    log_probs = torch.log(probs + 1e-8)  # (B, C)

    # H = -Σ p log p
    H = -torch.sum(probs * log_probs, dim=1)  # (B,)

    # E_p[z] = Σ p_c * z_c
    E_z = torch.sum(probs * logits, dim=1)  # (B,)

    # E_p[z log p] = Σ p_c * z_c * log p_c
    E_z_logp = torch.sum(probs * logits * log_probs, dim=1)  # (B,)

    # dH/dτ = (1/τ²) * [E_p[z log p] + H * E_p[z]]
    dH_dtau = (1 / (tau ** 2)) * (E_z_logp + H * E_z)  # (B,)

    return dH_dtau, H


@torch.no_grad()
def find_tau_safe_newton(logits, H_target, tau_min=0.1, tau_max=20.0,
                          max_iters=10, tol=1e-4):
    """
    Find τ such that H(softmax(logits/τ)) = H_target using Safe Newton method.

    Safe Newton: Use Newton step when valid, fallback to bisection otherwise.

    Args:
        logits: (B, C) - teacher logits
        H_target: (B,) - target entropy per sample
        tau_min, tau_max: search bounds
        max_iters: maximum iterations
        tol: convergence tolerance
    Returns:
        tau: (B,) - optimal temperature per sample
    """
    B = logits.size(0)
    device = logits.device

    # Initialize τ at midpoint
    tau = torch.full((B,), (tau_min + tau_max) / 2, device=device)

    # Track bounds for bisection fallback
    tau_lo = torch.full((B,), tau_min, device=device)
    tau_hi = torch.full((B,), tau_max, device=device)

    for _ in range(max_iters):
        # Compute f(τ) = H(τ) - H_target and f'(τ) = dH/dτ
        dH_dtau, H_current = entropy_derivative_wrt_tau(logits, tau)
        f = H_current - H_target  # (B,)
        f_prime = dH_dtau  # (B,)

        # Check convergence
        converged = torch.abs(f) < tol
        if converged.all():
            break

        # Newton step: τ_new = τ - f/f'
        # Avoid division by zero
        safe_f_prime = torch.where(torch.abs(f_prime) > 1e-10, f_prime,
                                    torch.sign(f_prime) * 1e-10 + 1e-10)
        tau_newton = tau - f / safe_f_prime

        # Update bisection bounds based on f sign
        # If f > 0: H_current > H_target, need smaller H, so decrease τ → tau_hi = tau
        # If f < 0: H_current < H_target, need larger H, so increase τ → tau_lo = tau
        tau_hi = torch.where(f > 0, tau, tau_hi)
        tau_lo = torch.where(f < 0, tau, tau_lo)

        # Safe Newton: use Newton step if within bounds, else bisection
        in_bounds = (tau_newton > tau_lo) & (tau_newton < tau_hi)
        tau_bisection = (tau_lo + tau_hi) / 2

        tau_new = torch.where(in_bounds & ~converged, tau_newton, tau_bisection)
        tau = torch.where(converged, tau, tau_new)

        # Clamp to bounds
        tau = torch.clamp(tau, tau_min, tau_max)

    return tau


def train_one_epoch(student, teacher, student_ref, train_loader, optimizer, device,
                    alpha, epoch, tau_min=0.1, tau_max=20.0):
    """Train student for one epoch with sample-wise entropy-matched KD."""
    student.train()
    teacher.eval()
    student_ref.eval()

    total_loss = 0.0
    total_ce = 0.0
    total_kl = 0.0
    total_tau = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for images, targets in pbar:
        images, targets = images.to(device), targets.to(device)

        # Forward pass
        student_logits = student(images)
        with torch.no_grad():
            teacher_logits = teacher(images)
            student_ref_logits = student_ref(images)

        # Compute target entropy from student_ref (per sample)
        student_ref_probs = F.softmax(student_ref_logits, dim=1)
        H_target = entropy(student_ref_probs)  # (B,)

        # Find τ per sample via Safe Newton
        tau = find_tau_safe_newton(teacher_logits, H_target,
                                    tau_min=tau_min, tau_max=tau_max)  # (B,)

        # CE loss
        ce_loss = F.cross_entropy(student_logits, targets)

        # KL loss with sample-wise τ (no T^2 scaling!)
        soft_teacher = F.softmax(teacher_logits / tau.unsqueeze(1), dim=1)
        soft_student = F.log_softmax(student_logits, dim=1)  # T=1 for student
        kl_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean')

        # Combined loss
        loss = (1 - alpha) * ce_loss + alpha * kl_loss

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Stats
        total_loss += loss.item()
        total_ce += ce_loss.item()
        total_kl += kl_loss.item()
        total_tau += tau.mean().item()
        _, predicted = student_logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'tau': f'{tau.mean().item():.2f}',
            'acc': f'{100.*correct/total:.1f}%'
        })

    n_batches = len(train_loader)
    return {
        'loss': total_loss / n_batches,
        'ce': total_ce / n_batches,
        'kl': total_kl / n_batches,
        'tau': total_tau / n_batches,
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


@hydra.main(config_path="distill/conf", config_name="entropy_matched_kd", version_base=None)
def main(cfg: DictConfig):
    logger.info("=" * 60)
    logger.info("Entropy-Matched KD (Sample-wise, Safe Newton)")
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
    # Load pretrained models
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Loading pretrained models")
    logger.info("=" * 60)

    # Load teacher (ResNet56)
    teacher = resnet56(num_classes=cfg.model.num_classes).to(device)
    teacher_ckpt = torch.load(cfg.kd.teacher_checkpoint, map_location=device)
    teacher.load_state_dict(teacher_ckpt)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    logger.info(f"Loaded teacher from {cfg.kd.teacher_checkpoint}")

    # Load reference student (pretrained ResNet20 for entropy target)
    student_ref = resnet20(num_classes=cfg.model.num_classes).to(device)
    student_ref_ckpt = torch.load(cfg.kd.student_ref_checkpoint, map_location=device)
    student_ref.load_state_dict(student_ref_ckpt)
    student_ref.eval()
    for param in student_ref.parameters():
        param.requires_grad = False
    logger.info(f"Loaded reference student from {cfg.kd.student_ref_checkpoint}")

    # Evaluate pretrained models
    teacher_acc = evaluate(teacher, test_loader, device)
    student_ref_acc = evaluate(student_ref, test_loader, device)
    logger.info(f"Teacher test accuracy: {teacher_acc:.2f}%")
    logger.info(f"Student ref test accuracy: {student_ref_acc:.2f}%")

    # =========================================================================
    # Train student with sample-wise entropy-matched KD
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Training with sample-wise entropy-matched KD")
    logger.info("=" * 60)

    # Initialize fresh student
    student = resnet20(num_classes=cfg.model.num_classes).to(device)

    optimizer = optim.SGD(
        student.parameters(),
        lr=cfg.train.lr,
        momentum=cfg.train.momentum,
        weight_decay=cfg.train.weight_decay
    )

    # Learning rate scheduler (step decay at 150, 180, 210)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 180, 210], gamma=0.1)

    best_acc = 0.0
    alpha = cfg.kd.alpha

    logger.info(f"\nTraining config:")
    logger.info(f"  Alpha (KL weight): {alpha}")
    logger.info(f"  Epochs: {cfg.train.epochs}")
    logger.info(f"  LR: {cfg.train.lr}")
    logger.info(f"  tau range: [{cfg.kd.t_min}, {cfg.kd.t_max}]")
    logger.info(f"  Sample-wise tau via Safe Newton (no T^2 scaling)")

    for epoch in range(1, cfg.train.epochs + 1):
        # Train
        train_stats = train_one_epoch(
            student, teacher, student_ref, train_loader, optimizer, device,
            alpha=alpha, epoch=epoch,
            tau_min=cfg.kd.t_min, tau_max=cfg.kd.t_max
        )

        # Evaluate
        test_acc = evaluate(student, test_loader, device)

        # Update best
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(student.state_dict(), 'best_student.pt')

        # Step scheduler
        scheduler.step()

        logger.info(
            f"Epoch {epoch}/{cfg.train.epochs} | "
            f"Loss: {train_stats['loss']:.4f} | "
            f"CE: {train_stats['ce']:.4f} | "
            f"KL: {train_stats['kl']:.4f} | "
            f"Tau: {train_stats['tau']:.2f} | "
            f"Train: {train_stats['acc']:.2f}% | "
            f"Test: {test_acc:.2f}% | "
            f"Best: {best_acc:.2f}% | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    logger.info(f"Best Test Accuracy: {best_acc:.2f}%")
    logger.info(f"Teacher Accuracy: {teacher_acc:.2f}%")
    logger.info(f"Student Ref Accuracy: {student_ref_acc:.2f}%")


if __name__ == "__main__":
    main()
