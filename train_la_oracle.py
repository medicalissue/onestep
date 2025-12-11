"""
Lookahead Oracle Self-Distillation (LA-Oracle)
===============================================

Core Idea:
  Teacher 없이, lookahead 기반으로 oracle distribution을 정의하여 self-distillation.

Mathematical Framework:
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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def entropy(probs):
    """Compute entropy of probability distributions. probs: (B, C) -> (B,)"""
    return -torch.sum(probs * torch.log(probs + 1e-8), dim=1)


def compute_entropy_gradient(probs):
    """
    Compute gradient of entropy w.r.t. logits.
    
    ∂H/∂z = p·(log p + 1 + H)
    
    Args:
        probs: (B, C) softmax probabilities
    Returns:
        g_H: (B, C) gradient of H w.r.t. logits
    """
    log_probs = torch.log(probs + 1e-8)
    H = -torch.sum(probs * log_probs, dim=1, keepdim=True)  # (B, 1)
    g_H = probs * (log_probs + 1 + H)  # (B, C)
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


def compute_lookahead_logits(logits, probs, targets, features, eta=1.0, weight_decay=0.0):
    """
    Compute exact lookahead logits using analytical gradient for last linear layer.
    
    For z = W·h + b with weight decay λ_wd:
      Full gradient: ∇CE + λ_wd·θ
      
      W⁺ = W - η·((p - e_y)⊗h + λ_wd·W)
      b⁺ = b - η·((p - e_y) + λ_wd·b)
      
      z⁺ = W⁺·h + b⁺ = z - η·(p - e_y)·(||h||² + 1) - η·λ_wd·z
    
    Thus:
      Δz = -η·(p - e_y)·(||h||² + 1) - η·λ_wd·z
      z̃_oracle = z + Δz
    
    Args:
        logits: (B, C) current logits
        probs: (B, C) softmax probabilities
        targets: (B,) ground truth indices
        features: (B, D) pre-fc features
        eta: learning rate (ODE time step)
        weight_decay: weight decay coefficient (0 to disable)
    Returns:
        z_oracle: (B, C) lookahead oracle logits
        delta_z: (B, C) logit change
    """
    B, C = logits.size()
    
    # CE gradient w.r.t. logits
    g_CE = compute_ce_gradient(probs, targets)  # (B, C)
    
    # Feature norm squared + 1 (for bias term)
    h_norm_sq = torch.sum(features ** 2, dim=1, keepdim=True)  # (B, 1)
    scale = h_norm_sq + 1.0  # (B, 1)
    
    # Δz from CE: -η·(p - e_y)·(||h||² + 1)
    delta_z_ce = -eta * g_CE * scale  # (B, C)
    
    # Δz from weight decay: -η·λ_wd·z (shrinks logits towards 0)
    delta_z_wd = -eta * weight_decay * logits  # (B, C)
    
    # Total Δz
    delta_z = delta_z_ce + delta_z_wd
    
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
        # dH/dτ = Var(z_oracle/τ) / τ² = (E[z²] - E[z]²) / τ³
        z_scaled = z_oracle / tau_expanded
        E_z = torch.sum(q * z_scaled, dim=1)
        E_z2 = torch.sum(q * z_scaled ** 2, dim=1)
        var_z = E_z2 - E_z ** 2
        dH_dtau = var_z / (tau ** 2 + 1e-10)
        
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


def la_oracle_loss(logits, features, targets, cfg, current_lambda, current_lr, delta_h_ema=None):
    """
    Compute Lookahead Oracle Self-Distillation loss.
    
    ODE Interpretation:
      Oracle q* is the CE ODE solution integrated forward by time Δt = current_lr.
      This is the natural time unit of one optimizer step.
    
    L(θ) = CE + λ·w_i·KL(q* || p_θ)
    
    Enhancements:
      A. Sample weighting: w_i = clamp(|ΔH_pred,i| / EMA(|ΔH_pred|), w_min, w_max)
         - Uses lookahead entropy change as difficulty measure
      B. H* gap amplification: ΔH_goal *= delta_h_alpha
    
    Note: η (lookahead time) = current_lr. No separate lookahead_scale needed.
          Regularization strength is controlled by λ, delta_h_alpha, w_i.
    
    Args:
        logits: (B, C) model logits
        features: (B, D) pre-fc features
        targets: (B,) ground truth indices
        cfg: config with hyperparameters
        current_lambda: KL weight (may be scheduled)
        current_lr: current learning rate (= ODE time step Δt)
        delta_h_ema: EMA of |ΔH_pred| for sample weighting (optional)
    Returns:
        total_loss, ce_loss, kl_loss, diagnostics, updated delta_h_ema
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
    
    # 1. Current probabilities
    probs = F.softmax(logits, dim=1)
    
    # 2. CE Loss per sample
    ce_per_sample = F.cross_entropy(logits, targets, reduction='none')  # (B,)
    ce_loss = ce_per_sample.mean()
    
    # 3. Lookahead computation
    with torch.no_grad():
        # 3a. Lookahead logits (exact, analytical) - compute Δz first
        z_oracle, delta_z = compute_lookahead_logits(
            logits, probs, targets, features, eta=eta, weight_decay=weight_decay
        )
        
        # 3b. Entropy change using actual Δz (consistent with logit change)
        delta_H_pred, H_current = compute_lookahead_entropy_change(probs, delta_z)
        
        # Difficulty metric: |ΔH_pred| - how much optimizer wants to move this sample's entropy
        abs_delta_H = torch.abs(delta_H_pred)  # (B,)
        abs_delta_H_mean = abs_delta_H.mean()
        
        # Update EMA of |ΔH_pred| for sample weighting
        if delta_h_ema is None:
            delta_h_ema_new = abs_delta_H_mean
        else:
            delta_h_ema_new = ema_decay * delta_h_ema + (1 - ema_decay) * abs_delta_H_mean
        
        # H* gap amplification (controls "how much to follow future")
        delta_H_goal = delta_h_alpha * delta_H_pred
        
        # Oracle entropy target
        H_target = H_current + delta_H_goal
        H_target = torch.clamp(H_target, 0.0, math.log(C))  # Valid range
        
        # 3c. Find τ* to match oracle entropy
        tau_star, q_star = find_tau_star(
            z_oracle, H_target,
            tau_min=tau_min, tau_max=tau_max, n_iters=bs_iters
        )
    
    # 6. KL Loss per sample: KL(q* || p_θ)
    log_probs = F.log_softmax(logits, dim=1)
    # KL per sample: sum over classes
    kl_per_sample = torch.sum(q_star * (torch.log(q_star + 1e-8) - log_probs), dim=1)  # (B,)
    
    # Sample-adaptive λ based on difficulty |ΔH_pred|
    # d_i = 1 (average) → λ_i = 0.5
    # d_i = 2+ (hard)   → λ_i = 1.0 (all KL, no CE)
    # d_i = 0 (easy)    → λ_i = 0.0 (all CE, no KL)
    if use_sample_weight:
        d_i = abs_delta_H / (delta_h_ema_new + 1e-6)  # centered at 1
        lambda_i = torch.clamp(d_i / 2.0, 0.0, 1.0)  # d=1→λ=0.5, d=2→λ=1, d=0→λ=0
        
        # Per-sample convex loss: (1-λ_i)*CE_i + λ_i*KL_i
        loss_per_sample = (1 - lambda_i) * ce_per_sample + lambda_i * kl_per_sample
        total_loss = loss_per_sample.mean()
        avg_weight = lambda_i.mean()  # average λ for logging
    else:
        # No sample weighting: use global λ
        total_loss = (1 - current_lambda) * ce_loss + current_lambda * kl_per_sample.mean()
        avg_weight = torch.tensor(current_lambda)
        delta_h_ema_new = None
    
    # Diagnostics
    with torch.no_grad():
        delta_z_norm = torch.norm(delta_z, dim=1).mean()
    
    return total_loss, ce_loss, kl_per_sample.mean(), H_current.mean(), H_target.mean(), tau_star.mean(), delta_z_norm, avg_weight, delta_h_ema_new


@hydra.main(version_base=None, config_path="distill/conf", config_name="la_oracle")
def main(cfg: DictConfig):
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
    
    logger.info("Starting Training...")
    start_time = time.time()
    
    best_acc = 0.0
    delta_h_ema = None  # EMA of |ΔH_pred| for sample weighting
    
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
            
            loss, ce, kl, h_cur, h_tgt, tau, dz_norm, w_avg, delta_h_ema = la_oracle_loss(
                logits, features, targets, cfg, current_lambda, current_lr, delta_h_ema
            )
            
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
