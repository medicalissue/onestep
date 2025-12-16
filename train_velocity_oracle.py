"""
Learn From Your Future: Velocity Oracle Distillation

Key idea: Learn a velocity predictor v_φ(h, z) that predicts logit dynamics,
then use K-step lookahead as oracle for self-distillation.

v_φ(h, z) ≈ E[Δz | h, z] = -η(softmax(z) - E[e_y | h, z])

Fixes:
1. v_φ takes (h, z) not just z - features help identify samples
2. Delayed supervision normalized by T_epoch for scale matching
"""

import os
import math
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from models import resnet20

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed=42):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class VelocityPredictor(nn.Module):
    """
    MLP to predict logit velocity: v_φ(h, z) ≈ Δz
    
    Input: 
        h (B, D) - backbone features (identifies sample)
        z (B, C) - current logits (current state)
    Output: 
        v (B, C) - predicted velocity (logit change per step)
    """
    def __init__(self, feat_dim=64, num_classes=100, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, h, z):
        # h: (B, feat_dim), z: (B, num_classes)
        x = torch.cat([h, z], dim=1)
        return self.net(x)


def compute_virtual_step(logits, targets, lr):
    """
    Compute virtual step: Δz_virtual = -η(p - e_y)
    
    This is the immediate gradient-based logit change.
    """
    probs = F.softmax(logits, dim=1)
    B, C = logits.size()
    e_y = torch.zeros_like(logits)
    e_y.scatter_(1, targets.view(-1, 1), 1.0)
    
    delta_z = -lr * (probs - e_y)
    return delta_z


def k_step_oracle(velocity_predictor, h_init, z_init, K):
    """
    Predict K-step lookahead using velocity predictor.
    
    ẑ^(k+1) = ẑ^(k) + v_φ(h, ẑ^(k))
    
    Note: h is fixed (same sample), only z evolves
    
    Returns: z after K steps, oracle distribution q*
    """
    z = z_init.detach()
    h = h_init.detach()
    for _ in range(K):
        v = velocity_predictor(h, z)
        z = z + v
    
    q_star = F.softmax(z, dim=1)
    return z, q_star


@hydra.main(version_base=None, config_path="distill/conf", config_name="velocity_oracle")
def main(cfg: DictConfig):
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set seed
    set_seed(cfg.train.get("seed", 42))
    logger.info("Seed set for reproducibility")
    
    # Data
    logger.info("Preparing Data...")
    
    if cfg.data.get("use_augmentation", True):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    trainloader = DataLoader(
        trainset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=4
    )
    testloader = DataLoader(
        testset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=4
    )
    
    num_classes = 100
    N = len(trainset)
    
    # Models
    logger.info("Creating Models...")
    net = resnet20(num_classes=num_classes).to(device)
    
    # Get feature dimension from model
    feat_dim = net.fc.in_features  # 64 for ResNet20
    
    velocity_predictor = VelocityPredictor(
        feat_dim=feat_dim,
        num_classes=num_classes,
        hidden_dim=cfg.velocity.get("hidden_dim", 256)
    ).to(device)
    
    # Optimizers
    optimizer_main = optim.SGD(
        net.parameters(),
        lr=cfg.train.lr,
        momentum=cfg.train.momentum,
        weight_decay=cfg.train.weight_decay
    )
    
    optimizer_vel = optim.Adam(
        velocity_predictor.parameters(),
        lr=cfg.velocity.get("lr", 1e-3)
    )
    
    # LR scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_main, T_max=cfg.train.epochs
    )
    
    # Hyperparameters
    K = cfg.velocity.get("K", 5)  # Lookahead steps
    lambda_kl = cfg.velocity.get("lambda_kl", 0.5)  # Distillation weight
    beta_delayed = cfg.velocity.get("beta_delayed", 1.0)  # Delayed loss weight
    
    # Logit and feature buffers for delayed supervision
    logit_buffer_prev = torch.zeros(N, num_classes, device=device)
    logit_buffer_curr = torch.zeros(N, num_classes, device=device)
    feat_buffer_prev = torch.zeros(N, feat_dim, device=device)
    feat_buffer_curr = torch.zeros(N, feat_dim, device=device)
    buffer_filled = False
    steps_per_epoch = len(trainloader)  # T_epoch for normalization
    
    logger.info(f"K-step lookahead: {K}")
    logger.info(f"Lambda (KL weight): {lambda_kl}")
    logger.info("Starting Training...")
    
    # Training loop
    for epoch in range(cfg.train.epochs):
        net.train()
        velocity_predictor.train()
        
        train_loss = 0
        train_ce = 0
        train_kl = 0
        train_vel = 0
        correct = 0
        total = 0
        
        pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{cfg.train.epochs}")
        
        # We need indices for buffer - use enumerate trick
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            B = inputs.size(0)
            
            # Compute global indices for buffer
            start_idx = batch_idx * cfg.train.batch_size
            indices = torch.arange(start_idx, min(start_idx + B, N), device=device)
            if len(indices) < B:
                indices = torch.arange(start_idx, start_idx + B, device=device) % N
            
            # Get current learning rate
            current_lr = optimizer_main.param_groups[0]['lr']
            
            # 1. Forward pass (main model) - get features too
            logits, features = net.forward_with_features(inputs)
            probs = F.softmax(logits, dim=1)
            
            # 2. Compute virtual step (online supervision)
            with torch.no_grad():
                delta_z_virtual = compute_virtual_step(logits, targets, current_lr)
            
            # 3. Update velocity predictor (online supervision)
            # Use cosine similarity (direction) + magnitude loss to prevent collapse to zero
            v_pred = velocity_predictor(features.detach(), logits.detach())
            
            # Cosine similarity loss (1 - cos_sim = 0 when aligned)
            cos_sim = F.cosine_similarity(v_pred, delta_z_virtual, dim=1).mean()
            loss_direction = 1 - cos_sim
            
            # Magnitude loss (log scale to handle small values better)
            v_norm = v_pred.norm(dim=1) + 1e-8
            target_norm = delta_z_virtual.norm(dim=1) + 1e-8
            loss_magnitude = F.mse_loss(torch.log(v_norm), torch.log(target_norm))
            
            # Combined loss: direction + magnitude
            loss_vel_online = loss_direction + 0.1 * loss_magnitude
            
            optimizer_vel.zero_grad()
            loss_vel_online.backward()
            optimizer_vel.step()
            
            # 4. K-step oracle prediction
            with torch.no_grad():
                z_oracle, q_star = k_step_oracle(velocity_predictor, features, logits, K)
            
            # 5. Main model loss
            loss_ce = F.cross_entropy(logits, targets)
            loss_kl = F.kl_div(
                F.log_softmax(logits, dim=1),
                q_star.detach(),
                reduction='batchmean'
            )
            
            loss_main = loss_ce + lambda_kl * loss_kl
            
            optimizer_main.zero_grad()
            loss_main.backward()
            optimizer_main.step()
            
            # 6. Save current logits and features to buffer
            logit_buffer_curr[indices[:len(logits)]] = logits.detach()
            feat_buffer_curr[indices[:len(features)]] = features.detach()
            
            # Stats
            train_loss += loss_main.item()
            train_ce += loss_ce.item()
            train_kl += loss_kl.item()
            train_vel += loss_vel_online.item()
            
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{loss_main.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%',
                'VelLoss': f'{loss_vel_online.item():.4f}'
            })
            
            # Diagnostic logging every 100 batches
            if batch_idx % 100 == 0:
                with torch.no_grad():
                    dz_norm = delta_z_virtual.norm(dim=1).mean()
                    v_norm = v_pred.norm(dim=1).mean()
                    z_diff = (z_oracle - logits.detach()).norm(dim=1).mean()
                    # Also check max values
                    dz_max = delta_z_virtual.abs().max()
                    v_max = v_pred.abs().max()
                logger.info(f"[Batch {batch_idx}] dz_norm: {dz_norm:.4f}, v_norm: {v_norm:.4f}, "
                           f"z_diff: {z_diff:.4f}, dz_max: {dz_max:.4f}, v_max: {v_max:.4f}, KL: {loss_kl.item():.6f}")
            
            # Dry run
            if cfg.velocity.get("dry_run", False):
                logger.info(f"Dry Run | Loss: {loss_main.item():.4f} | CE: {loss_ce.item():.4f} | "
                           f"KL: {loss_kl.item():.4f} | VelLoss: {loss_vel_online.item():.4f}")
                logger.info("Dry run complete.")
                return
        
        # End of epoch: Delayed supervision
        if buffer_filled:
            logger.info("Performing delayed calibration...")
            delta_z_real = logit_buffer_curr - logit_buffer_prev
            
            # Scale normalization: T_epoch steps occurred
            # Predictor learns per-step velocity, so divide by T
            delta_z_normalized = delta_z_real / steps_per_epoch
            
            # Mini-batch delayed training
            indices_all = torch.randperm(N, device=device)
            delayed_loss_total = 0
            n_delayed_batches = 0
            
            for i in range(0, N, cfg.train.batch_size):
                batch_indices = indices_all[i:i + cfg.train.batch_size]
                h_prev = feat_buffer_prev[batch_indices]
                z_prev = logit_buffer_prev[batch_indices]
                dz_norm = delta_z_normalized[batch_indices]
                
                v_pred = velocity_predictor(h_prev, z_prev)
                
                # Same loss as online: cosine + magnitude
                cos_sim = F.cosine_similarity(v_pred, dz_norm, dim=1).mean()
                loss_dir = 1 - cos_sim
                v_norm = v_pred.norm(dim=1) + 1e-8
                target_norm = dz_norm.norm(dim=1) + 1e-8
                loss_mag = F.mse_loss(torch.log(v_norm), torch.log(target_norm))
                loss_delayed = loss_dir + 0.1 * loss_mag
                
                optimizer_vel.zero_grad()
                (beta_delayed * loss_delayed).backward()
                optimizer_vel.step()
                
                delayed_loss_total += loss_delayed.item()
                n_delayed_batches += 1
            
            avg_delayed_loss = delayed_loss_total / n_delayed_batches
            logger.info(f"Delayed calibration done. Avg loss: {avg_delayed_loss:.4f}")
        
        # Update buffers
        logit_buffer_prev = logit_buffer_curr.clone()
        feat_buffer_prev = feat_buffer_curr.clone()
        buffer_filled = True
        
        # Scheduler step
        scheduler.step()
        
        # Epoch stats
        train_loss /= len(trainloader)
        train_ce /= len(trainloader)
        train_kl /= len(trainloader)
        train_vel /= len(trainloader)
        train_acc = 100. * correct / total
        
        # Evaluation
        net.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
        
        test_acc = 100. * test_correct / test_total
        
        logger.info(f"Epoch {epoch+1}/{cfg.train.epochs} | "
                   f"TrainLoss: {train_loss:.4f} | TrainAcc: {train_acc:.2f}% | "
                   f"TestAcc: {test_acc:.2f}% | CE: {train_ce:.4f} | KL: {train_kl:.4f} | "
                   f"VelLoss: {train_vel:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % cfg.train.get("save_every", 50) == 0:
            state = {
                'net': net.state_dict(),
                'velocity_predictor': velocity_predictor.state_dict(),
                'epoch': epoch,
                'test_acc': test_acc,
            }
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(state, f'checkpoints/velocity_oracle_epoch{epoch+1}.pth')
    
    logger.info("Training complete!")
    logger.info(f"Final Test Accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    main()
