import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import torch
import torch.optim as optim
import wandb
import math
import os
from tqdm import tqdm
from distill.data_loader_llm import get_llm_loaders
from distill.models_llm import get_llm_models
from distill.distillers import PRISM_LLM

log = logging.getLogger(__name__)

@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    # WandB Setup
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=f"LLM_{cfg.distill.method}_{cfg.wandb.name}",
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    
    device = torch.device(cfg.distill.solver_device if torch.cuda.is_available() else "cpu")
    
    # Output Dir
    output_dir = f"outputs/{cfg.model.student_name}_{cfg.distill.method}_{cfg.wandb.name}"
    os.makedirs(output_dir, exist_ok=True)

    # Data Loading
    log.info(f"Loading LLM Dataset: {cfg.data.dataset_name}...")
    
    loaders = get_llm_loaders(
        dataset_name=cfg.data.dataset_name,
        model_name=cfg.model.student_name, # Use student tokenizer
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        data_root='./data'
    )
    
    if len(loaders) == 4:
        trainloader, valloader, testloader, tokenizer = loaders
    else:
        # Fallback for old behavior (train, test, tokenizer)
        trainloader, testloader, tokenizer = loaders
        valloader = testloader # Use test as val

    
    # Model Setup
    log.info(f"Initializing Models (Teacher: {cfg.model.teacher_name}, Student: {cfg.model.student_name})...")
    teacher, student = get_llm_models(
        teacher_name=cfg.model.teacher_name,
        student_name=cfg.model.student_name,
        device=device,
        dropout=cfg.model.get("dropout", 0.1)
    )
    
    # Optimizer
    optimizer = optim.AdamW(
        student.parameters(), 
        lr=cfg.distill.lr, 
        weight_decay=cfg.distill.weight_decay
    )
    
    # Distiller
    log.info("Initializing PRISM_LLM...")
    distiller = PRISM_LLM(student, teacher, optimizer, device, cfg)
    
    # Training Loop
    epochs = cfg.distill.epochs
    log.info(f"Starting Training for {epochs} epochs...")
    
    best_val_ppl = float('inf')

    for epoch in range(epochs):
        student.train()
        total_loss = 0
        
        pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}")
        for i, batch in enumerate(pbar):
            inputs = batch["input_ids"]
            targets = batch["labels"] # DataCollatorForLM creates labels automatically
            
            # PRISM Step
            distiller.train_step(inputs, targets)
            
            # Logging
            loss = distiller.log_dict.get('loss', 0)
            total_loss += loss
            
            pbar.set_postfix(loss=loss, **{k: v for k, v in distiller.log_dict.items() if k != 'loss'})
            
            if i % 10 == 0:
                wandb.log(distiller.log_dict)
        
        avg_train_loss = total_loss / len(trainloader)
        train_ppl = math.exp(avg_train_loss)
        log.info(f"Epoch {epoch+1} Train PPL: {train_ppl:.2f}")
        
        # Validation
        student.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in valloader:
                inputs = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = student(inputs, labels=labels)
                total_val_loss += outputs.loss.item()
        
        avg_val_loss = total_val_loss / len(valloader)
        val_ppl = math.exp(avg_val_loss)
        
        log.info(f"Epoch {epoch+1} Val PPL: {val_ppl:.2f}")
        wandb.log({"val_ppl": val_ppl, "epoch": epoch+1})
        
        # Save Checkpoint (Per Epoch)
        ckpt_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save(student.state_dict(), ckpt_path)
        log.info(f"Saved checkpoint to {ckpt_path}")
        
        # Save Best Model
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            best_path = os.path.join(output_dir, "best_model.pth")
            torch.save(student.state_dict(), best_path)
            log.info(f"New Best Model (PPL {best_val_ppl:.2f}) saved to {best_path}")
            wandb.save(best_path)

    log.info("Training Complete.")

if __name__ == "__main__":
    main()
