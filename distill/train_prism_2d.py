import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
import math
import os
from tqdm import tqdm
from distill.data_loader import get_cifar_loaders
from distill.models_cifar import resnet20, resnet56, mobilenetv2
from distill.prism_2d import PRISM2D

log = logging.getLogger(__name__)

@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    # WandB Setup
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=f"PRISM2D_{cfg.wandb.name}",
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    
    device = torch.device(cfg.distill.solver_device if torch.cuda.is_available() else "cpu")
    
    # Output Dir
    output_dir = f"outputs/prism2d_{cfg.model.student_name}_{cfg.wandb.name}"
    os.makedirs(output_dir, exist_ok=True)

    # Data Loading
    log.info(f"Loading Dataset: {cfg.data.dataset_name}...")
    trainloader, testloader = get_cifar_loaders(
        dataset_name=cfg.data.dataset_name,
        data_root='./data', 
        batch_size=cfg.data.batch_size, 
        num_workers=cfg.data.num_workers
    )
    
    num_classes = 100 if cfg.data.dataset_name == 'cifar100' else 10

    # Model Setup
    log.info(f"Initializing Student: {cfg.model.student_name}, Teacher: {cfg.model.teacher_name}...")
    
    # Student
    if cfg.model.student_name == 'resnet20':
        student = resnet20(num_classes=num_classes).to(device)
    else:
        raise ValueError(f"Unsupported student: {cfg.model.student_name}")
        
    # Teacher (Standard ResNet56 or similar)
    if cfg.model.teacher_name == 'resnet56':
        teacher = resnet56(num_classes=num_classes).to(device)
        if hasattr(cfg.model, 'teacher_path') and cfg.model.teacher_path and cfg.model.teacher_path != "None":
            log.info(f"Loading Teacher from {cfg.model.teacher_path}...")
            teacher.load_state_dict(torch.load(cfg.model.teacher_path, map_location=device))
        else:
            log.warning("No teacher path provided! Using random teacher (BAD).")
    else:
        # Fallback
        teacher = resnet56(num_classes=num_classes).to(device)
        
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
        
    # Optimizer
    optimizer = optim.SGD(
        student.parameters(), 
        lr=cfg.distill.lr, 
        momentum=cfg.distill.get('momentum', 0.9),
        weight_decay=cfg.distill.weight_decay
    )
    
    # Scheduler
    def warmup_scheduler(epoch):
        if epoch < 5:
            return (epoch + 1) / 5.0
        else:
            return 1.0
            
    warmup_sched = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_scheduler)
    main_sched = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    
    # Distiller
    distiller = PRISM2D(student, teacher, optimizer, device, cfg)
    
    # Training Loop
    epochs = cfg.distill.epochs
    log.info(f"Starting 2D PRISM Training for {epochs} epochs...")
    
    best_acc = 0.0

    for epoch in range(epochs):
        student.train()
        total_loss = 0
        
        if epoch < 5:
            warmup_sched.step()
        else:
            main_sched.step()
            
        current_lr = optimizer.param_groups[0]['lr']
        
        pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs} [LR={current_lr:.4f}]")
        for i, (inputs, labels) in enumerate(pbar):
            # Distiller Step
            distiller.train_step(inputs, labels)
            
            # Logging
            log_dict = distiller.log_dict
            
            pbar.set_postfix(
                w_par=f"{log_dict['w_par']:.2f}", 
                w_perp=f"{log_dict['w_perp']:.2f}",
                beta_par=f"{log_dict['beta_par']:.2f}"
            )
            
            if i % 100 == 0:
                wandb.log(log_dict)
        
        # Validation
        student.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = student(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        acc = 100. * correct / total
        log.info(f"Epoch {epoch+1} Test Accuracy: {acc:.2f}%")
        wandb.log({"test_acc": acc, "epoch": epoch+1})
        
        # Save Checkpoint
        if acc > best_acc:
            best_acc = acc
            torch.save(student.state_dict(), os.path.join(output_dir, "best_model.pth"))
            log.info(f"New Best Model Saved (Acc {best_acc:.2f}%)")

if __name__ == "__main__":
    main()
