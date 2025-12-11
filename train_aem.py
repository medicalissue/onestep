"""
Adaptive Entropy Matching (AEM)
===============================

Core Idea:
  CE forces overconfidence on ALL samples.
  We add sample-adaptive entropy regularization.

Loss = CE + λ * (H(p) - H*)²

where:
  - H(p) = entropy of model prediction
  - H* = target entropy (adaptive, based on difficulty)
  - H* = h_min + (h_max - h_min) * d_i
  - d_i = difficulty estimate from CE loss
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

# Import ResNet20 from the existing codebase
from models import resnet20

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def entropy(probs):
    """
    Compute entropy of probability distributions.
    probs: (B, C)
    Returns: (B,)
    """
    return -torch.sum(probs * torch.log(probs + 1e-8), dim=1)


def H_epsilon_rest(eps, C):
    """
    Entropy of canonical symmetric REST distribution (non-GT classes only):
      p_rest(c) = 1/(C-1) for all c ≠ y  (uniform when eps doesn't apply here)
    
    For rest entropy, we use:
      H_rest_min = 0 (all mass on single wrong class)
      H_rest_max = log(C-1) (uniform over all wrong classes)
    
    But with epsilon interpretation for target:
      When noise rate is ε, the rest distribution has entropy ≈ log(k) where k = ε*(C-1)
    """
    if eps < 1e-7:
        return 0.0
    # Effective number of confused classes among C-1 wrong classes
    # When ε is the noise rate, assume roughly ε*(C-1) classes get significant mass
    k_eff = max(1.0, eps * (C - 1))
    return math.log(k_eff)


def rest_entropy(probs, targets):
    """
    Compute conditional entropy of non-GT classes.
    
    H_rest = H(p_rest) where p_rest(c) = p(c) / (1 - p(y)) for c ≠ y
    
    Args:
        probs: (B, C) softmax probabilities
        targets: (B,) ground truth indices
    Returns:
        (B,) rest entropy per sample
    """
    B, C = probs.size()
    
    # Get GT probabilities
    p_y = probs.gather(1, targets.view(-1, 1)).squeeze(1)  # (B,)
    
    # Compute rest mass: 1 - p_y
    rest_mass = (1 - p_y).clamp(min=1e-8)  # (B,)
    
    # Zero out GT class to get only non-GT probabilities
    mask = torch.ones_like(probs, dtype=torch.bool)
    mask.scatter_(1, targets.view(-1, 1), False)
    p_rest = probs.clone()
    p_rest[~mask] = 0  # Zero out GT
    
    # Normalize to get conditional distribution
    p_rest_cond = p_rest / rest_mass.view(-1, 1)  # (B, C) but GT entry is 0
    
    # Compute entropy (only over non-GT entries)
    h_rest = -torch.sum(p_rest_cond * torch.log(p_rest_cond + 1e-8), dim=1)
    
    return h_rest


def aem_loss(logits, targets, cfg, current_lambda):
    """
    Entropy Rate Regularizer.
    
    Loss = CE + λ * rate_penalty
    
    Key insight:
      - ΔH_pred = -η * ⟨∇H, ∇CE⟩ estimates how much CE will change H
      - Only penalize EXCESSIVE entropy reduction (ΔH_pred < -κ)
      - Allow entropy increase (overconfident wrong → correction OK)
    
    rate_penalty = relu(-κ - ΔH_pred)²
    
    Benefits:
      - Only 2 hyperparams: λ (strength), κ (threshold)
      - Sample-adaptive automatically (via gradient magnitudes)
      - No explicit difficulty score needed
      - Easy samples: small gradients → no constraint
      - Hard samples: large reduction → brake applied
    """
    B, C = logits.size()
    
    # 1. Compute softmax probabilities
    probs = F.softmax(logits, dim=1)
    
    # 2. CE gradient w.r.t. logits: g_CE = p - one_hot(y)
    one_hot_targets = F.one_hot(targets, num_classes=C).float()
    g_CE = probs - one_hot_targets  # (B, C)
    
    # 3. Entropy gradient w.r.t. logits
    # ∂H/∂z = ∂H/∂p * ∂p/∂z
    # For softmax: ∂p_i/∂z_j = p_i(δ_ij - p_j)
    # ∂H/∂p_i = -(1 + log p_i)
    # After chain rule: g_H_c = -p_c * (H + log p_c + 1 - Σ_j p_j log p_j) 
    # Simplified: g_H = p * (log p + 1) - p * (Σ p log p + 1) = p * log p - p * H
    log_probs = torch.log(probs + 1e-8)
    H = -torch.sum(probs * log_probs, dim=1, keepdim=True)  # (B, 1)
    g_H = probs * (log_probs + 1 + H)  # (B, C) - gradient of H w.r.t. logits
    
    # 4. Lookahead entropy change: ΔH_pred ≈ -η * ⟨g_H, g_CE⟩
    # We absorb η into κ, so just compute the inner product
    inner_product = torch.sum(g_H * g_CE, dim=1)  # (B,)
    delta_H_pred = -inner_product  # (B,) - predicted change in H per step
    
    # 5. Rate penalty: only penalize excessive reduction
    # If ΔH_pred < -κ: reducing entropy too fast → penalty
    # If ΔH_pred >= -κ: OK (including increase)
    kappa = cfg.aem.get("kappa", 0.1)  # Maximum allowed entropy reduction
    
    # penalty = relu(-κ - ΔH_pred)²
    # = relu(-κ + inner_product)²  since ΔH_pred = -inner_product
    excess = torch.relu(-kappa - delta_H_pred)
    rate_penalty = excess ** 2
    
    # 6. CE Loss
    ce_per_sample = F.cross_entropy(logits, targets, reduction='none')
    ce_loss = ce_per_sample.mean()
    
    # 7. Total Loss
    rate_loss_mean = rate_penalty.mean()
    total_loss = ce_loss + current_lambda * rate_loss_mean
    
    # For logging
    with torch.no_grad():
        frac_penalized = (delta_H_pred < -kappa).float().mean()
        avg_delta = delta_H_pred.mean()
    
    return total_loss, ce_loss, rate_loss_mean, avg_delta, frac_penalized, H.squeeze().mean()


@hydra.main(version_base=None, config_path="distill/conf", config_name="aem")
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
    
    for epoch in range(cfg.train.epochs):
        # Lambda Schedule
        lambda_schedule = cfg.aem.get("lambda_schedule", "cosine")
        lambda_max = cfg.aem.lambda_val
        
        if lambda_schedule == "static":
            current_lambda = lambda_max
        elif lambda_schedule == "cosine":
            # Cosine increasing: 0 → lambda_max
            progress = epoch / cfg.train.epochs
            current_lambda = lambda_max * (math.sin(math.pi / 2 * progress) ** 2)
        else:
            current_lambda = lambda_max
        
        net.train()
        train_loss = 0
        train_ce = 0
        train_ent = 0
        correct = 0
        total = 0
        
        avg_diff = 0
        avg_h_tgt = 0
        avg_h_model = 0
        num_batches = 0
        
        pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{cfg.train.epochs}")
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            logits = net(inputs)
            
            loss, ce, rate_loss, avg_delta, frac_pen, h_model = aem_loss(logits, targets, cfg, current_lambda)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_ce += ce.item()
            train_ent += rate_loss.item()
            
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            avg_diff += avg_delta.item()
            avg_h_tgt += frac_pen.item()
            avg_h_model += h_model.item()
            num_batches += 1
            
            pbar.set_postfix({
                'Loss': f"{train_loss/num_batches:.4f}",
                'Acc': f"{100.*correct/total:.2f}%",
                'H': f"{avg_h_model/num_batches:.2f}",
                'Pen%': f"{100.*avg_h_tgt/num_batches:.0f}%",
                'λ': f"{current_lambda:.3f}"
            })
            
            if cfg.aem.get("dry_run", False):
                logger.info(f"Dry Run | Loss: {loss.item():.4f} | CE: {ce.item():.4f} | "
                           f"Rate: {rate_loss.item():.4f} | ΔH: {avg_delta.item():.3f} | "
                           f"Penalized: {100.*frac_pen.item():.0f}% | H: {h_model.item():.2f}")
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
                    f"H: {avg_h_model/num_batches:.2f} | Penalized: {100.*avg_h_tgt/num_batches:.0f}% | "
                    f"λ: {current_lambda:.3f}")

    total_time = time.time() - start_time
    logger.info(f"Training Finished. Best Acc: {best_acc:.2f}% | Time: {total_time/60:.2f} mins")


if __name__ == '__main__':
    main()
