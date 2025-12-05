import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import logging
import wandb
import os
import math

from distill.models import ResNetWrapper, SimpleCNN

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    # 1. Setup WandB
    if cfg.wandb.mode != "disabled":
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=f"PRISM_beta{cfg.distill.beta}_eta{cfg.distill.eta1}_{cfg.distill.eta2}",
            config=OmegaConf.to_container(cfg, resolve=True)
        )
    
    # 2. Setup Data
    log.info("Loading CIFAR-10...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=cfg.data.num_workers)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers)
    
    # 3. Setup Models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if hasattr(cfg.distill, 'teacher_path') and cfg.distill.teacher_path:
        log.info(f"Loading Teacher from local path: {cfg.distill.teacher_path}...")
        # Initialize with pretrained=False since we load custom weights
        teacher = ResNetWrapper(num_classes=10, pretrained=False).to(device)
        teacher.load_state_dict(torch.load(cfg.distill.teacher_path, map_location=device))
    else:
        log.info("Initializing Teacher (ResNet18) from HF: edadaltocg/resnet18_cifar10...")
        teacher = ResNetWrapper(num_classes=10, pretrained=True, repo_id="edadaltocg/resnet18_cifar10").to(device)
        
    teacher.eval() # Freeze Teacher
    
    # Disable gradient tracking for teacher weights
    for param in teacher.parameters():
        param.requires_grad = False
        
    log.info("Initializing Student (SimpleCNN)...")
    student = SimpleCNN(num_classes=10).to(device)
    student.train()
    
    # 4. Optimizer
    optimizer = optim.Adam(student.parameters(), lr=1e-3, weight_decay=cfg.distill.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.distill.epochs)
    
    # 5. PRISM Training Loop
    beta = cfg.distill.beta
    eta1 = cfg.distill.eta1
    eta2 = cfg.distill.eta2
    epochs = cfg.distill.epochs
    
    # Max Entropy for 10 classes = ln(10)
    H_max = math.log(10)
    
    log.info(f"Starting PRISM Training (Epochs: {epochs}, Beta: {beta})...")
    
    for epoch in range(epochs):
        student.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for i, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # --- PRISM-D Step ---
            
            # 1. Forward Passes
            with torch.no_grad():
                t_logits = teacher(inputs)
                t_probs = F.softmax(t_logits, dim=1)
                
                # Soft Correctness (Alpha)
                # alpha = p_target (Probability assigned to GT class)
                # Gather probabilities of target classes
                alpha_per_sample = t_probs.gather(1, targets.view(-1, 1)).squeeze()
                alpha_batch = alpha_per_sample.mean().item()
                
                # Entropy (Just for logging)
                entropy = -torch.sum(t_probs * torch.log(t_probs + 1e-8), dim=1)
                avg_entropy = entropy.mean().item()

            s_logits = student(inputs)
            
            # 2. Compute Soft Gradient (g_s)
            T = 4.0
            loss_kl = nn.KLDivLoss(reduction="batchmean")(
                F.log_softmax(s_logits / T, dim=1),
                F.softmax(t_logits / T, dim=1)
            ) * (T * T)
            
            optimizer.zero_grad()
            loss_kl.backward(retain_graph=True)
            
            grad_s = {}
            for name, param in student.named_parameters():
                if param.grad is not None:
                    grad_s[name] = param.grad.clone()
                    param.grad = None 
            
            # 3. Compute Hard Gradient (g_h)
            loss_ce = F.cross_entropy(s_logits, targets)
            loss_ce.backward()
            
            # 4. Gradient Projection & Combination (PRISM-D)
            for name, param in student.named_parameters():
                if param.grad is not None and name in grad_s:
                    g_h = param.grad 
                    g_s = grad_s[name]
                    
                    g_h_flat = g_h.view(-1)
                    g_s_flat = g_s.view(-1)
                    
                    # Unit Vector Projection (Stability)
                    norm_s = torch.norm(g_s_flat)
                    g_s_hat = g_s_flat / (norm_s + 1e-8)
                    
                    dot = torch.dot(g_h_flat, g_s_hat)
                    proj = dot * g_s_hat
                    proj = proj.view_as(g_h)
                    
                    # PRISM-D Update Rule
                    # g_final = alpha * g_s + g_h - (alpha if dot < 0 else 0) * proj
                    
                    # 1. Base: Scaled Soft Gradient
                    g_final = alpha_batch * g_s
                    
                    # 2. Hard Gradient Component
                    if dot >= 0:
                        # Synergy: Trust Hard Gradient fully
                        g_final += g_h
                    else:
                        # Conflict: Remove conflicting component based on correctness
                        # If alpha=1 (Trust Teacher): Remove proj (g_h_perp)
                        # If alpha=0 (Trust Hard): Keep proj (g_h)
                        g_final += g_h - alpha_batch * proj
                    
                    param.grad = g_final
            
            optimizer.step()
            
            # Metrics
            total_loss += loss_ce.item() 
            
            _, predicted = s_logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if i % 100 == 0:
                log.info(f"Epoch [{epoch+1}/{epochs}] Batch [{i}] Loss: {loss_ce.item():.4f} Acc: {100.*correct/total:.2f}% Alpha(Soft): {alpha_batch:.4f} Entropy: {avg_entropy:.4f}")
        
        scheduler.step()
        
        # --- Validation ---
        student.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = student(inputs)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
        
        acc = 100. * test_correct / test_total
        log.info(f"Epoch {epoch+1} Test Accuracy: {acc:.2f}%")
        
        if cfg.wandb.mode != "disabled":
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": total_loss / len(trainloader),
                "test_acc": acc,
                "avg_lambda": avg_lambda
            })
            
    log.info("PRISM Training Complete.")
    if cfg.wandb.mode != "disabled":
        wandb.finish()

if __name__ == "__main__":
    main()
