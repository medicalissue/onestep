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

class LossEMA:
    def __init__(self, decay=0.99):
        self.decay = decay
        self.min_loss = torch.tensor(0.0)
        self.max_loss = torch.tensor(5.0) # Initial guess
        self.initialized = False
        
    def update(self, loss_batch):
        with torch.no_grad():
            batch_min = loss_batch.min()
            batch_max = loss_batch.max()
            
            if not self.initialized:
                self.min_loss = batch_min
                self.max_loss = batch_max
                self.initialized = True
            else:
                self.min_loss = self.decay * self.min_loss + (1 - self.decay) * batch_min
                self.max_loss = self.decay * self.max_loss + (1 - self.decay) * batch_max

    def get_range(self):
        return self.min_loss, self.max_loss

def la_e_rls_loss(logits, targets, cfg, device, loss_ema, current_lambda):
    """
    Compute the LA-E-RLS loss.
    """
    B, C = logits.size()
    
    # 1. Hard CE Loss & Difficulty Estimation
    ce_loss_per_sample = F.cross_entropy(logits, targets, reduction='none')
    
    # Update EMA
    loss_ema.update(ce_loss_per_sample.cpu())
    l_min, l_max = loss_ema.get_range()
    l_min = l_min.to(device)
    l_max = l_max.to(device)
    
    with torch.no_grad():
        # Difficulty score d_i (Version B: Entropy-based)
        # s_{i,y} = exp(-ell_i)
        s_iy = torch.exp(-ce_loss_per_sample)
        
        # Avoid numerical instability for log(1-s_iy) when s_iy is close to 1
        # 1 - s_iy could be 0.
        one_minus_s = 1.0 - s_iy
        one_minus_s = torch.clamp(one_minus_s, min=1e-8)
        
        # H_tilde = -s_iy * log(s_iy) - (1-s_iy) * log((1-s_iy)/(C-1))
        # Note: log(s_iy) = -ce_loss_per_sample
        term1 = s_iy * ce_loss_per_sample
        term2 = one_minus_s * (torch.log(one_minus_s) - math.log(C - 1))
        
        h_tilde = term1 - term2
        
        # d_i = H_tilde / log(C)
        d_i = h_tilde / math.log(C)
        d_i = torch.clamp(d_i, 0.0, 1.0)
        
        # 2. Target Entropy
        h_target = cfg.la_e_rls.h_min + (cfg.la_e_rls.h_max - cfg.la_e_rls.h_min) * d_i
        
        # 3. Base RLS Distribution p_i
        # Adaptive alpha based on difficulty (Version C)
        # alpha_i = alpha_min + (alpha_max - alpha_min) * d_i
        alpha = cfg.la_e_rls.alpha_min + (cfg.la_e_rls.alpha_max - cfg.la_e_rls.alpha_min) * d_i
        
        p_i = torch.zeros(B, C, device=device)
        p_i.scatter_(1, targets.view(-1, 1), (1 - alpha).view(-1, 1))
        
        distribution_type = cfg.la_e_rls.get("distribution_type", "power_law")
        
        if distribution_type == "dirichlet":
            # Dirichlet Distribution (The "Sexy" Option)
            # Sample from symmetric Dirichlet with concentration beta
            beta_val = cfg.la_e_rls.get("dirichlet_beta", 0.0)
            
            # Auto-scaling: If beta <= 0, set to 1/C to maintain constant sparsity
            if beta_val <= 0:
                beta_val = 1.0 / C
            
            # Create concentration tensor (B, C)
            concentration = torch.full((B, C), beta_val, device=device)
            m = torch.distributions.Dirichlet(concentration)
            r = m.sample()
        else:
            # Power-law noise (Original)
            # r ~ Uniform(0, 1)^k -> pushes most values towards 0, leaving few high values
            r = torch.rand(B, C, device=device).pow(cfg.la_e_rls.power_law_exp)

        # Zero out the target index in the noise term so we can explicitly control GT prob with alpha
        r.scatter_(1, targets.view(-1, 1), 0.0)
        
        r_sum = r.sum(dim=1, keepdim=True) + 1e-8
        r_norm = r / r_sum
        
        p_i += r_norm * alpha.view(-1, 1)
        
        # 4. Hybrid Secant/Bisection Method (Brent's Method approximation)
        # Combines robustness of Bisection with speed of Newton/Secant.
        
        tau_min = torch.full((B,), cfg.la_e_rls.tau_min, device=device)
        tau_max = torch.full((B,), cfg.la_e_rls.tau_max, device=device)
        tau = (tau_min + tau_max) / 2.0
        
        # Logits for gradient calculation
        z = torch.log(p_i + 1e-10)
        
        for _ in range(cfg.la_e_rls.bs_iters):
            tau_expanded = tau.view(-1, 1)
            
            # Forward
            logits_scaled = z / tau_expanded
            q = F.softmax(logits_scaled, dim=1)
            h_cur = entropy(q)
            diff = h_cur - h_target
            
            # Update Bracket (Bisection Logic)
            # If H > H_target (Too flat) -> Need smaller Tau -> High = Tau
            # If H < H_target (Too sharp) -> Need larger Tau -> Low = Tau
            # Note: H is monotonically increasing with Tau.
            mask_too_soft = diff > 0
            tau_max = torch.where(mask_too_soft, tau, tau_max)
            tau_min = torch.where(~mask_too_soft, tau, tau_min)
            
            # Gradient Calculation (for Secant/Newton step)
            E_z = (q * z).sum(dim=1)
            E_z2 = (q * z.pow(2)).sum(dim=1)
            var_z = E_z2 - E_z.pow(2)
            dH_dtau = var_z / (tau.pow(3) + 1e-10)
            
            # Secant/Newton Step
            # tau_new = tau - diff / H'
            update = diff / (dH_dtau + 1e-10)
            tau_secant = tau - update
            
            # Check Validity of Secant Step
            # Must be strictly within the CURRENT bracket [tau_min, tau_max]
            # We add a small buffer to avoid getting stuck at edges
            buffer = (tau_max - tau_min) * 0.1
            is_secant_valid = (tau_secant > tau_min + buffer) & (tau_secant < tau_max - buffer)
            
            # Fallback to Bisection
            tau_bisection = (tau_min + tau_max) / 2.0
            
            # Select Next Tau
            tau = torch.where(is_secant_valid, tau_secant, tau_bisection)
            
        tau_star = tau
        
        # Virtual Teacher Logits
        # z_t = log(p_i) / tau_star
        log_p_i = torch.log(p_i + 1e-10)
        tau_kd = cfg.la_e_rls.tau_kd # Moved here to be available for z_t calculation
        z_t = log_p_i / (tau_star.view(-1, 1) * tau_kd)
        q_star = temperature_transform(p_i, tau_star.view(-1, 1))
        
    # 5. Soft KL Loss (Permutation Invariant / Sorted KL)
    # Distillation with fixed KD temperature (tau_kd)
    
    # Teacher target: q_distill
    total_tau = tau_star.view(-1, 1) * tau_kd
    q_distill = temperature_transform(p_i, total_tau)
    
    # Student prediction: p_s
    # logits are raw logits, so we apply temperature scaling directly
    p_s = F.softmax(logits / tau_kd, dim=1)
    
    sorting_mode = cfg.la_e_rls.get("sorting_mode", "fully_sorted")
    
    if sorting_mode == "gt_anchored":
        # GT-Anchored Sorted KL (Correct Implementation)
        # Goal: Incentive Student GT to be the Max, matching Teacher's Max.
        # Teacher: Fully Sorted (Index 0 is Max).
        # Student: Anchor GT at Index 0, Sort Rest at Index 1..
        
        # 1. Teacher Side: Just Sort (Permutation Invariant)
        # q[0] will be the Maximum probability (which is effectively the Target prob in our generation logic)
        q_distill_sorted, _ = torch.sort(q_distill, descending=True, dim=1)
        
        # 2. Student Side: Anchor GT
        
        # A. Extract GT prob
        target_indices = targets.view(-1, 1)
        p_s_gt = p_s.gather(1, target_indices) # (B, 1)
        
        # B. Sort Rest
        # Clone to avoid modifying original
        p_s_temp = p_s.clone()
        # Set GT to -1.0 (so it moves to the end after sort)
        p_s_temp.scatter_(1, target_indices, -1.0)
        # Sort descending
        p_s_sorted_all, _ = torch.sort(p_s_temp, descending=True, dim=1)
        # Slice off the last column (the GT we set to -1.0)
        p_s_non_gt_sorted = p_s_sorted_all[:, :-1]
        
        # C. Concat: [GT, Sorted_Rest]
        # Now Index 0 is GT. Index 1.. are sorted non-GT.
        p_s_sorted = torch.cat([p_s_gt, p_s_non_gt_sorted], dim=1)
        
        # Result:
        # KL( p_s_sorted || q_distill_sorted )
        # p_s_sorted[0] (GT) <-> q_distill_sorted[0] (Max)
        # p_s_sorted[1..] (Rest) <-> q_distill_sorted[1..] (Rest)
        
    else:
        # Fully Sorted KL (Original)
        p_s_sorted, _ = torch.sort(p_s, descending=True, dim=1)
        q_distill_sorted, _ = torch.sort(q_distill, descending=True, dim=1)
        
        if torch.isnan(p_s_sorted).any() or torch.isnan(q_distill_sorted).any():
            print("NaN detected in sorted probs!")
            print("p_s_sorted:", p_s_sorted)
            print("q_distill_sorted:", q_distill_sorted)
        

    
    # KL Divergence
    # Add epsilon to avoid log(0)
    kl_loss = F.kl_div((p_s_sorted + 1e-10).log(), q_distill_sorted, reduction='batchmean')
    
    # Scale by tau_kd^2
    kl_loss = kl_loss * (tau_kd ** 2)
    
    # Adaptive Lambda Schedule
    lambda_schedule = cfg.la_e_rls.get("lambda_schedule", "cosine")
    
    if lambda_schedule == "adaptive":
        # Adaptive: Lambda is proportional to mastery (1 - difficulty)
        # Mastery increases -> Lambda increases (More regularization)
        # Difficulty increases -> Lambda decreases (More focus on CE)
        batch_mastery = 1.0 - d_i.mean()
        # Clamp mastery to [0, 1] just in case
        batch_mastery = torch.clamp(batch_mastery, 0.0, 1.0)
        
        current_lambda = cfg.la_e_rls.lambda_val * batch_mastery
    
    # If lambda_schedule is "cosine", current_lambda is passed from outside
    
    ce_loss = ce_loss_per_sample.mean()
    total_loss = (1 - current_lambda) * ce_loss + current_lambda * kl_loss
    
    return total_loss, ce_loss, kl_loss, d_i.mean(), h_target.mean(), tau_star.mean(), current_lambda

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
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 180, 210], gamma=0.1)
    
    logger.info("Starting Training...")
    start_time = time.time()
    
    loss_ema = LossEMA()
    
    for epoch in range(cfg.train.epochs):
        # Cosine Increasing Schedule for Lambda
        # lambda(t) = lambda_max * sin^2(pi/2 * t / T)
        lambda_max = cfg.la_e_rls.lambda_val
        progress = epoch / cfg.train.epochs
        current_lambda = lambda_max * (math.sin(math.pi / 2 * progress) ** 2)
        
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
            
            loss, ce, kl, d_mean, h_mean, tau_mean, lambda_used = la_e_rls_loss(logits, targets, cfg, device, loss_ema, current_lambda)
            
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
