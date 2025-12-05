import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
import os
from distill.data_loader import get_cifar_loaders
from distill.models import ResNetWrapper, SimpleCNN
from distill.models_cifar import resnet20, resnet56, mobilenetv2
from distill.distillers import KD, PRISM, PCGrad, DKD

log = logging.getLogger(__name__)

@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    # WandB Setup
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=f"{cfg.distill.method}_{cfg.wandb.name}",
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    
    # Data Loading
    trainloader, testloader = get_cifar_loaders(
        dataset_name=cfg.data.dataset_name,
        data_root='./data', 
        batch_size=cfg.data.batch_size, 
        num_workers=cfg.data.num_workers
    )
    
    # Model Setup
    device = torch.device(cfg.distill.solver_device if torch.cuda.is_available() else "cpu")
    num_classes = 100 if cfg.data.dataset_name == 'cifar100' else 10
    
    # Teacher
    if cfg.model.teacher_name == 'resnet56':
        teacher = resnet56(num_classes=num_classes).to(device)
        if hasattr(cfg.model, 'teacher_path') and cfg.model.teacher_path and cfg.model.teacher_path != "None":
            log.info(f"Loading Teacher from local path: {cfg.model.teacher_path}...")
            teacher.load_state_dict(torch.load(cfg.model.teacher_path, map_location=device))
        else:
            log.warning("No teacher path provided for ResNet56. Using random weights (NOT RECOMMENDED for distillation).")
    else:
        # Fallback to existing logic
        if hasattr(cfg.model, 'teacher_path') and cfg.model.teacher_path and cfg.model.teacher_path != "None":
            log.info(f"Loading Teacher from local path: {cfg.model.teacher_path}...")
            teacher = ResNetWrapper(num_classes=num_classes, pretrained=False).to(device)
            teacher.load_state_dict(torch.load(cfg.model.teacher_path, map_location=device))
        else:
            log.info("Initializing Teacher (ResNet18) from HF: edadaltocg/resnet18_cifar10...")
            teacher = ResNetWrapper(num_classes=num_classes, pretrained=True, repo_id="edadaltocg/resnet18_cifar10").to(device)
    
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
        
    # Student
    if cfg.model.student_name == 'resnet20':
        log.info(f"Initializing Student (ResNet20) for {num_classes} classes...")
        student = resnet20(num_classes=num_classes).to(device)
    elif cfg.model.student_name == 'mobilenetv2':
        log.info(f"Initializing Student (MobileNetV2) for {num_classes} classes...")
        student = mobilenetv2(num_classes=num_classes).to(device)
    else:
        log.info("Initializing Student (SimpleCNN)...")
        student = SimpleCNN(num_classes=num_classes).to(device)
    
    # Optimizer
    optimizer = optim.SGD(
        student.parameters(), 
        lr=cfg.distill.lr, 
        momentum=cfg.distill.get('momentum', 0.9),
        weight_decay=cfg.distill.weight_decay
    )
    # Scheduler: Linear Warmup (20) + MultiStepLR (150, 180, 210)
    def warmup_scheduler(epoch):
        if epoch < 20:
            return (epoch + 1) / 20.0
        else:
            return 1.0
            
    warmup_sched = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_scheduler)
    main_sched = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 180, 210], gamma=0.1)
    
    # Select Distiller
    log.info(f"Initializing Distiller: {cfg.distill.method.upper()}...")
    
    if cfg.distill.method == 'kd':
        distiller = KD(student, teacher, optimizer, device, cfg)
    elif cfg.distill.method == 'prism':
        distiller = PRISM(student, teacher, optimizer, device, cfg)
    elif cfg.distill.method == 'pcgrad':
        distiller = PCGrad(student, teacher, optimizer, device, cfg)
    elif cfg.distill.method == 'dkd':
        distiller = DKD(student, teacher, optimizer, device, cfg)
    else:
        raise ValueError(f"Unknown method: {cfg.distill.method}")
    
    # Training Loop
    epochs = cfg.distill.epochs
    log.info(f"Starting Training for {epochs} epochs...")
    
    from tqdm import tqdm
    
    for epoch in range(epochs):
        student.train()
        total_loss = 0
        # Update Scheduler
        if epoch < 20:
            warmup_sched.step()
        else:
            main_sched.step()
            
        current_lr = optimizer.param_groups[0]['lr']
        
        pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs} [LR={current_lr:.4f}]")
        for i, (inputs, targets) in enumerate(pbar):
            # inputs, targets are moved to device inside train_step
            
            # Distiller Step
            distiller.train_step(inputs, targets)
            
            # Logging
            loss = distiller.log_dict.get('loss', 0)
            total_loss += loss
            
            pbar.set_postfix(loss=loss, **{k: v for k, v in distiller.log_dict.items() if k != 'loss'})
            
            if i % 100 == 0:
                wandb.log(distiller.log_dict)
        
        # Validation
        student.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = student(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        acc = 100. * correct / total
        log.info(f"Epoch {epoch+1} Test Accuracy: {acc:.2f}%")
        wandb.log({"test_acc": acc, "epoch": epoch+1, "lr": current_lr})

    log.info("Training Complete.")

if __name__ == "__main__":
    main()
