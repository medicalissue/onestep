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
from distill.data_loader_llm import get_llm_loaders
from distill.models_llm import get_llm_models
from distill.models_pruning import PruningLLM
from distill.ema import ModelEMA

log = logging.getLogger(__name__)

@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    # WandB Setup
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=f"Pruning_{cfg.wandb.name}",
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    
    device = torch.device(cfg.distill.solver_device if torch.cuda.is_available() else "cpu")
    
    # Output Dir
    output_dir = f"outputs/pruning_{cfg.model.student_name}_{cfg.wandb.name}"
    os.makedirs(output_dir, exist_ok=True)

    # Data Loading
    log.info(f"Loading LLM Dataset: {cfg.data.dataset_name}...")
    loaders = get_llm_loaders(
        dataset_name=cfg.data.dataset_name,
        model_name=cfg.model.student_name,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        data_root='./data'
    )
    
    if len(loaders) == 4:
        trainloader, valloader, testloader, tokenizer = loaders
    else:
        trainloader, testloader, tokenizer = loaders
        valloader = testloader

    # Model Setup
    log.info(f"Initializing Student Model: {cfg.model.student_name}...")
    # We only need student model initially. Teacher is EMA of student.
    # But get_llm_models returns both. We can use it or just load student.
    # Let's use get_llm_models to be consistent with config, but ignore teacher return if we want self-distill only.
    # Actually, the user specified "No external teacher". So we just load student.
    
    from transformers import AutoModelForCausalLM, AutoConfig
    student_config = AutoConfig.from_pretrained(cfg.model.student_name)
    student_config.resid_pdrop = cfg.model.get("dropout", 0.1)
    student_config.embd_pdrop = cfg.model.get("dropout", 0.1)
    student_config.attn_pdrop = cfg.model.get("dropout", 0.1)
    
    student_base = AutoModelForCausalLM.from_pretrained(cfg.model.student_name, config=student_config)
    student_base.to(device)
    student_base.train()
    
    # Wrap with PruningLLM
    student = PruningLLM(student_base)
    student.to(device)
    
    # EMA Teacher
    ema_decay = cfg.distill.get('ema_decay', 0.999)
    teacher = ModelEMA(student_base, decay=ema_decay, device=device)
    
    # Optimizer
    optimizer = optim.AdamW(
        student.parameters(), 
        lr=cfg.distill.lr, 
        weight_decay=cfg.distill.weight_decay
    )
    
    # Hyperparameters
    alpha = cfg.distill.get('alpha', 1.0) # KD weight
    lambda_sparse = cfg.distill.get('lambda_sparse', 0.01) # Sparsity weight
    T = cfg.distill.temperature
    
    # Training Loop
    epochs = cfg.distill.epochs
    log.info(f"Starting Activation-based Sparse Distillation for {epochs} epochs...")
    
    best_val_ppl = float('inf')

    for epoch in range(epochs):
        student.train() # PruningLLM forwards to student.train()
        total_loss = 0
        
        pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}")
        for i, batch in enumerate(pbar):
            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            # 1. Student Forward
            outputs = student(inputs, labels=labels)
            logits_student = outputs.logits
            loss_ce = outputs.loss
            
            # 2. Teacher Forward (EMA)
            with torch.no_grad():
                outputs_teacher = teacher(inputs)
                logits_teacher = outputs_teacher.logits
            
            # 3. KD Loss (Self-Distillation)
            loss_kd = nn.KLDivLoss(reduction="batchmean")(
                F.log_softmax(logits_student / T, dim=-1),
                F.softmax(logits_teacher / T, dim=-1)
            ) * (T * T)
            
            # 4. Sparsity Loss
            loss_sparse = student.get_sparsity_loss()
            
            # Total Loss
            loss = loss_ce + alpha * loss_kd + lambda_sparse * loss_sparse
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update EMA Teacher
            teacher.update(student_base)
            
            # Logging
            total_loss += loss.item()
            
            log_dict = {
                "loss": loss.item(),
                "loss_ce": loss_ce.item(),
                "loss_kd": loss_kd.item(),
                "loss_sparse": loss_sparse.item()
            }
            
            pbar.set_postfix(**log_dict)
            if i % 10 == 0:
                wandb.log(log_dict)
        
        # Validation
        avg_train_loss = total_loss / len(trainloader)
        log.info(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")
        
        student_base.eval() # Eval on base model
        total_val_loss = 0
        with torch.no_grad():
            for batch in valloader:
                inputs = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                outputs = student_base(inputs, labels=labels)
                total_val_loss += outputs.loss.item()
        
        val_ppl = math.exp(total_val_loss / len(valloader))
        log.info(f"Epoch {epoch+1} Val PPL: {val_ppl:.2f}")
        wandb.log({"val_ppl": val_ppl, "epoch": epoch+1})
        
        # Save Checkpoint
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            torch.save(student_base.state_dict(), os.path.join(output_dir, "best_model.pth"))
            log.info(f"New Best Model Saved (PPL {best_val_ppl:.2f})")

if __name__ == "__main__":
    main()
