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
from distill.models_cifar import resnet20

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def entropy(probs):
    """
    Compute entropy of a batch of probability distributions.
    probs: (B, C)
    Returns: (B,)
    """
    return -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

def temperature_transform(probs, tau):
    """
    Apply temperature scaling to probabilities.
    probs: (B, C)
    tau: (B, 1) or scalar
    """
    log_probs = torch.log(probs + 1e-10)
    scaled_logits = log_probs / tau
    return F.softmax(scaled_logits, dim=1)

def la_e_rls_loss(logits, targets, cfg, device):
    """
    Compute the LA-E-RLS loss.
    """
    B, C = logits.size()
    
    # 1. Hard CE Loss & Difficulty Estimation
    ce_loss_per_sample = F.cross_entropy(logits, targets, reduction='none')
    
    with torch.no_grad():
        l_min = ce_loss_per_sample.min()
        l_max = ce_loss_per_sample.max()
        
        # Difficulty score d_i in [0, 1]
        denom = l_max - l_min + 1e-8
        d_i = torch.clamp((ce_loss_per_sample - l_min) / denom, 0.0, 1.0)
        
        # 2. Target Entropy
        h_target = cfg.la_e_rls.h_min + (cfg.la_e_rls.h_max - cfg.la_e_rls.h_min) * d_i
        
        # 3. Base RLS Distribution p_i
        alpha = torch.empty(B, device=device).uniform_(cfg.la_e_rls.alpha_min, cfg.la_e_rls.alpha_max)
        
        p_i = torch.zeros(B, C, device=device)
        p_i.scatter_(1, targets.view(-1, 1), (1 - alpha).view(-1, 1))
        
        r = torch.rand(B, C, device=device)
        r.scatter_(1, targets.view(-1, 1), 0.0)
        
        r_sum = r.sum(dim=1, keepdim=True) + 1e-8
        r_norm = r / r_sum
        
        p_i += r_norm * alpha.view(-1, 1)
        
        # 4. Binary Search for Tau*
        tau_min = torch.full((B,), cfg.la_e_rls.tau_min, device=device)
        tau_max = torch.full((B,), cfg.la_e_rls.tau_max, device=device)
        
        for _ in range(cfg.la_e_rls.bs_iters):
            tau_mid = (tau_min + tau_max) / 2.0
            tau_mid_expanded = tau_mid.view(-1, 1)
            
            q_mid = temperature_transform(p_i, tau_mid_expanded)
            h_cur = entropy(q_mid)
            
            mask_too_soft = h_cur > h_target
            
            tau_max = torch.where(mask_too_soft, tau_mid, tau_max)
            tau_min = torch.where(~mask_too_soft, tau_mid, tau_min)
            
        tau_star = (tau_min + tau_max) / 2.0
        q_star = temperature_transform(p_i, tau_star.view(-1, 1))
        
    # 5. Soft KL Loss
    s_logits_tau = logits / cfg.la_e_rls.tau_s
    s_log_probs = F.log_softmax(s_logits_tau, dim=1)
    
    kl_loss = F.kl_div(s_log_probs, q_star, reduction='batchmean') * (cfg.la_e_rls.tau_s ** 2)
    
    ce_loss = ce_loss_per_sample.mean()
    total_loss = (1 - cfg.la_e_rls.lambda_val) * ce_loss + cfg.la_e_rls.lambda_val * kl_loss
    
    return total_loss, ce_loss, kl_loss, d_i.mean(), h_target.mean(), tau_star.mean()

@hydra.main(version_base=None, config_path="distill/conf", config_name="la_e_rls")
def main(cfg: DictConfig):
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data
    logger.info("Preparing Data...")
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
    
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=cfg.data.num_workers)
    
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers)
    
    # Model
    logger.info("Creating Model (ResNet20)...")
    net = resnet20(num_classes=100).to(device)
    
    optimizer = optim.SGD(net.parameters(), lr=cfg.train.lr, momentum=cfg.train.momentum, weight_decay=cfg.train.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    
    logger.info("Starting Training...")
    start_time = time.time()
    
    for epoch in range(cfg.train.epochs):
        net.train()
        train_loss = 0
        train_ce = 0
        train_kl = 0
        correct = 0
        total = 0
        
        avg_diff = 0
        avg_h_tgt = 0
        avg_tau = 0
        num_batches = 0
        
        pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{cfg.train.epochs}")
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            logits = net(inputs)
            
            loss, ce, kl, d_mean, h_mean, tau_mean = la_e_rls_loss(logits, targets, cfg, device)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_ce += ce.item()
            train_kl += kl.item()
            
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            avg_diff += d_mean.item()
            avg_h_tgt += h_mean.item()
            avg_tau += tau_mean.item()
            num_batches += 1
            
            # Update tqdm bar with running averages
            pbar.set_postfix({
                'Loss': f"{train_loss/num_batches:.4f}",
                'Acc': f"{100.*correct/total:.2f}%",
                'Diff': f"{avg_diff/num_batches:.2f}",
                'Tau': f"{avg_tau/num_batches:.2f}"
            })
            
            if cfg.la_e_rls.dry_run:
                logger.info(f"Dry Run Batch | Loss: {loss.item():.4f} | Diff: {d_mean.item():.4f} | H_tgt: {h_mean.item():.4f} | Tau: {tau_mean.item():.4f}")
                logger.info("Dry run complete.")
                return

        scheduler.step()
        
        # Validation
        net.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, targets in tqdm(testloader, desc="Validation", leave=False):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = F.cross_entropy(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
        
        acc = 100. * test_correct / test_total
        train_acc = 100. * correct / total
        
        logger.info(f"Epoch {epoch+1}/{cfg.train.epochs} Summary | "
                    f"Loss: {train_loss/num_batches:.4f} | "
                    f"Acc: {train_acc:.2f}% | Test Acc: {acc:.2f}%")

    total_time = time.time() - start_time
    logger.info(f"Training Finished. Total Time: {total_time/60:.2f} mins")

if __name__ == '__main__':
    main()
