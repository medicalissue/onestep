import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
from distill.data_loader_llm import get_llm_loaders
from distill.models_llm import get_llm_models
from distill.distillers import PRISM_LLM
import os

@hydra.main(config_path="distill/conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    # Override for analysis
    cfg.data.dataset_name = "dolly-15k"
    cfg.model.teacher_name = "gpt2-xl"
    cfg.model.student_name = "gpt2"
    cfg.distill.method = "prism"
    cfg.data.batch_size = 4
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Data
    print("Loading Data...")
    loaders = get_llm_loaders(
        dataset_name=cfg.data.dataset_name,
        model_name=cfg.model.student_name,
        batch_size=cfg.data.batch_size,
        num_workers=4,
        data_root='/data/dolly15k'
    )
    if len(loaders) == 4:
        trainloader, _, _, _ = loaders
    else:
        trainloader, _, _ = loaders
    
    # Load Models
    print("Loading Models...")
    teacher, student = get_llm_models(
        teacher_name=cfg.model.teacher_name,
        student_name=cfg.model.student_name,
        device=device
    )
    
    # Optimizer (Dummy)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-5)
    
    # Distiller
    distiller = PRISM_LLM(student, teacher, optimizer, device, cfg)
    
    # Logs
    logs = {
        'dist_L0': [], 'cos_L0': [],
        'dist_L6': [], 'cos_L6': [],
        'dist_L11': [], 'cos_L11': [],
        'avg_p_t_gt': [],
    }
    
    print("Running Analysis Loop (100 batches)...")
    student.train()
    
    for i, batch in enumerate(tqdm(trainloader, total=100)):
        if i >= 100:
            break
            
        inputs = batch["input_ids"]
        targets = batch["labels"]
        
        # Run step (computes gradients and metrics)
        distiller.train_step(inputs, targets)
        
        # Collect metrics
        logs['dist_L0'].append(distiller.log_dict['dist_L0'])
        logs['dist_L6'].append(distiller.log_dict['dist_L6'])
        logs['dist_L11'].append(distiller.log_dict['dist_L11'])
        logs['cos_L0'].append(distiller.log_dict['cos_L0'])
        logs['cos_L6'].append(distiller.log_dict['cos_L6'])
        logs['cos_L11'].append(distiller.log_dict['cos_L11'])
        logs['avg_p_t_gt'].append(distiller.log_dict['avg_p_t_gt'])
        
    # Plotting
    print("Plotting results...")
    plt.figure(figsize=(18, 5))
    
    # Plot 1: Layer 0 (Early)
    plt.subplot(1, 3, 1)
    plt.scatter(logs['dist_L0'], logs['avg_p_t_gt'], alpha=0.6, color='blue')
    corr_L0 = np.corrcoef(logs['dist_L0'], logs['avg_p_t_gt'])[0, 1]
    plt.xlabel('Gradient Distance (L0)')
    plt.ylabel('Teacher Prob P_T(y_gt)')
    plt.title(f'Early Layer (L0)\nCorr: {corr_L0:.3f}')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Layer 6 (Middle)
    plt.subplot(1, 3, 2)
    plt.scatter(logs['dist_L6'], logs['avg_p_t_gt'], alpha=0.6, color='green')
    corr_L6 = np.corrcoef(logs['dist_L6'], logs['avg_p_t_gt'])[0, 1]
    plt.xlabel('Gradient Distance (L6)')
    plt.ylabel('Teacher Prob P_T(y_gt)')
    plt.title(f'Middle Layer (L6)\nCorr: {corr_L6:.3f}')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Layer 11 (Late)
    plt.subplot(1, 3, 3)
    plt.scatter(logs['dist_L11'], logs['avg_p_t_gt'], alpha=0.6, color='red')
    corr_L11 = np.corrcoef(logs['dist_L11'], logs['avg_p_t_gt'])[0, 1]
    plt.xlabel('Gradient Distance (L11)')
    plt.ylabel('Teacher Prob P_T(y_gt)')
    plt.title(f'Late Layer (L11)\nCorr: {corr_L11:.3f}')
    plt.grid(True, alpha=0.3)
    
    save_path = "/home/junesang/.gemini/antigravity/brain/a8e7af0b-7b8d-4b1e-8732-8c3cdc43cf24/llm_layer_dist_hypothesis.png"
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    
    # Print Summary
    print(f"Analysis Complete.")
    print(f"Corr L0 (Dist vs Correctness): {corr_L0:.4f}")
    print(f"Corr L6 (Dist vs Correctness): {corr_L6:.4f}")
    print(f"Corr L11 (Dist vs Correctness): {corr_L11:.4f}")

if __name__ == "__main__":
    main()
