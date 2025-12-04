import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import wandb
import logging
from .models import SimpleMLP, RandomFeatureAdapter, ResNetWrapper, SimpleCNN
from .utils import compute_statistics
from .solver import OnePassDistiller

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    # 1. Setup WandB
    if cfg.wandb.mode != "disabled":
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.name,
            config=OmegaConf.to_container(cfg, resolve=True)
        )
    
    # 2. Setup Data & Models
    if cfg.data.dataset_name == "cifar10":
        log.info("Loading CIFAR-10...")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        # We use train set for distillation
        dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=cfg.data.num_workers)
        
        # Models
        # Teacher: Pretrained ResNet18
        log.info("Initializing Teacher (ResNet18)...")
        teacher = ResNetWrapper(num_classes=10, pretrained=True)
        teacher.eval() # Teacher is frozen
        
        # Student: Simple CNN
        log.info("Initializing Student (SimpleCNN)...")
        student_base = SimpleCNN(num_classes=10)
        
        # Update hidden dims for adapter
        # ResNet18 penultimate dim is 512
        teacher_hidden_dim = 512 
        # SimpleCNN penultimate dim is 2048 (128*4*4)
        student_hidden_dim = 2048
        
    else:
        # Synthetic
        log.info("Generating synthetic data...")
        X = torch.randn(cfg.data.num_samples, cfg.data.input_dim)
        dataset = torch.utils.data.TensorDataset(X)
        loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.data.batch_size, shuffle=True)
        
        teacher = SimpleMLP(cfg.data.input_dim, cfg.model.teacher_hidden_dim, cfg.model.output_dim)
        student_base = SimpleMLP(cfg.data.input_dim, cfg.model.student_hidden_dim, cfg.model.output_dim)
        
        teacher_hidden_dim = cfg.model.teacher_hidden_dim
        student_hidden_dim = cfg.model.student_hidden_dim
        
        # Initialize teacher with random weights (it's already random)
        with torch.no_grad():
            # Just to ensure it works
            _ = teacher(X[:2])

    # 3. Initialize Adapter
    # We distill to match the teacher's output (logits) or features?
    # For this scaffold, let's assume we match the residuals of the output.
    # Residual = Teacher(x) - StudentBase(x)
    
    # Check dimensions
    # Student hidden dim -> Adapter -> Projection Dim -> Output Dim
    
    adapter = RandomFeatureAdapter(
        input_dim=student_hidden_dim, # Adapter takes student's hidden state
        projection_dim=cfg.distill.projection_dim,
        activation=cfg.distill.activation,
        init_scale=cfg.distill.init_scale,
        init_method=cfg.distill.init_method,
        scales=cfg.distill.scales,
        depth=cfg.distill.adapter_depth
    )
    
    # Move models to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher.to(device)
    student_base.to(device)
    adapter.to(device)
    
    log.info(f"Starting One-Pass Distillation on {device}...")
    
    # Determine target dimension
    if cfg.distill.use_feature_distillation:
        target_dim = teacher_hidden_dim 
    else:
        target_dim = cfg.model.output_dim

    # Initialize Solver
    solver = OnePassDistiller(
        feature_dim=cfg.distill.projection_dim,
        output_dim=target_dim,
        lambda_reg=cfg.distill.lambda_reg,
        device=device
    )
    

    
    # --- FREEZE & FIT PHASE ---
    log.info("Accumulating statistics...")
    for i, batch in enumerate(loader):
        if isinstance(batch, list) or isinstance(batch, tuple):
            if len(batch) == 2:
                x_batch, _ = batch # CIFAR returns (x, y)
            else:
                x_batch = batch[0] # Synthetic returns (x,)
        else:
            x_batch = batch
            
        x_batch = x_batch.to(device)
        
        with torch.no_grad():
            # 1. Compute Teacher Output
            if cfg.distill.use_feature_distillation:
                _, target = teacher(x_batch, return_hidden=True)
            else:
                target = teacher(x_batch)
            
            # 2. Compute Student Base Output & Hidden
            y_student, h_student = student_base(x_batch, return_hidden=True)
            
            # Data-dependent Init (First batch only)
            if i == 0:
                log.info("Initializing adapter with data statistics...")
                adapter.initialize_from_data(h_student)
            
            # 3. Compute Residual
            if cfg.distill.use_feature_distillation:
                # Target is hidden state
                _, student_features = student_base(x_batch, return_hidden=True)
                if target.shape[1] == student_features.shape[1]:
                    residual = target - student_features
                else:
                    # Dimensions differ, cannot subtract. 
                    # Assume we want to predict the teacher feature directly (Student Base contribution = 0)
                    residual = target
            else:
                # Target is logits
                residual = target - y_student
            
            # 4. Extract Random Features
            # Adapter forward returns features if V is None
            features = adapter(h_student)
            
            # 5. Accumulate
            solver.accumulate_batch(features, residual)
            
            if i % 10 == 0:
                log.info(f"Processed batch {i}/{len(loader)}")

    # Solve
    W_star = solver.solve()
    
    # Update Adapter
    adapter.V = nn.Parameter(W_star) # W_star is (d_out, D)
    
    # --- VERIFICATION ---
    log.info("Verifying...")
    total_mse = 0
    total_correct = 0
    count = 0
    
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, list) or isinstance(batch, tuple):
                if len(batch) == 2:
                    x_batch, _ = batch
                else:
                    x_batch = batch[0]
            else:
                x_batch = batch
                
            x_batch = x_batch.to(device)
            
            if cfg.distill.use_feature_distillation:
                _, target = teacher(x_batch, return_hidden=True)
                _, student_base_out = student_base(x_batch, return_hidden=True)
            else:
                target = teacher(x_batch)
                student_base_out = student_base(x_batch)
                
            _, h_student = student_base(x_batch, return_hidden=True)
            
            # Student with Adapter
            # f_S(x) = f_base(x) + Adapter(h(x))
            adapter_out = adapter(h_student) # Now includes V projection
            
            if cfg.distill.use_feature_distillation:
                if target.shape[1] == student_base_out.shape[1]:
                     y_final = student_base_out + adapter_out
                else:
                     y_final = adapter_out # Base contribution is 0
            else:
                y_final = student_base_out + adapter_out
            
            mse = F.mse_loss(y_final, target, reduction='sum')
            total_mse += mse.item()
            count += x_batch.shape[0]
            
            # Accuracy Check (only if target is logits)
            if not cfg.distill.use_feature_distillation:
                # y_final is logits
                pred_labels = y_final.argmax(dim=1)
                teacher_labels = target.argmax(dim=1)
                correct = (pred_labels == teacher_labels).sum().item()
                total_correct += correct

    avg_mse = total_mse / count
    log.info(f"Final MSE: {avg_mse:.6f}")
    
    if not cfg.distill.use_feature_distillation:
        accuracy = total_correct / count * 100
        log.info(f"Student-Teacher Agreement (Accuracy): {accuracy:.2f}%")
        if cfg.wandb.mode != "disabled":
            wandb.log({"accuracy": accuracy})
    
    if cfg.wandb.mode != "disabled":
        wandb.log({"final_mse": avg_mse})
        wandb.finish()

if __name__ == "__main__":
    main()
