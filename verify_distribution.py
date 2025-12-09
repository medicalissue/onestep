import torch
import torch.nn.functional as F
import math

def entropy(p):
    return -torch.sum(p * torch.log(p + 1e-10), dim=1)

def temperature_transform(p, tau):
    log_p = torch.log(p + 1e-10)
    scaled_logits = log_p / tau
    return F.softmax(scaled_logits, dim=1)

def verify():
    # Config parameters
    h_min = 0.5
    h_max = 1.5
    alpha_min = 0.05
    alpha_max = 0.30
    tau_min_val = 0.5
    tau_max_val = 3.0
    power_law_exp = 5.0
    tau_kd = 1.0
    bs_iters = 10
    
    B = 5
    C = 100
    
    # Simulate logits for 5 samples with varying difficulty
    # Sample 0: Very Easy (High confidence on GT)
    # Sample 4: Very Hard (Uniform-ish)
    logits = torch.randn(B, C)
    targets = torch.zeros(B, dtype=torch.long) # Dummy targets (class 0)
    
    # Manually set logits to simulate difficulty
    logits[0, 0] = 10.0 # Easy
    logits[1, 0] = 5.0
    logits[2, 0] = 3.0
    logits[3, 0] = 1.0
    logits[4, 0] = 0.0 # Hard
    
    ce_loss_per_sample = F.cross_entropy(logits, targets, reduction='none')
    
    # Difficulty Estimation (Version B)
    s_iy = torch.exp(-ce_loss_per_sample)
    one_minus_s = 1.0 - s_iy
    one_minus_s = torch.clamp(one_minus_s, min=1e-8)
    term1 = s_iy * ce_loss_per_sample
    term2 = one_minus_s * (torch.log(one_minus_s) - math.log(C - 1))
    h_tilde = term1 - term2
    d_i = h_tilde / math.log(C)
    d_i = torch.clamp(d_i, 0.0, 1.0)
    
    # Target Entropy
    h_target = h_min + (h_max - h_min) * d_i
    
    # Adaptive Alpha
    alpha = alpha_min + (alpha_max - alpha_min) * d_i
    
    # Base RLS (Power Law)
    p_i = torch.zeros(B, C)
    p_i.scatter_(1, targets.view(-1, 1), (1 - alpha).view(-1, 1))
    
    r = torch.rand(B, C).pow(power_law_exp)
    r.scatter_(1, targets.view(-1, 1), 0.0)
    r_sum = r.sum(dim=1, keepdim=True) + 1e-8
    r_norm = r / r_sum
    p_i += r_norm * alpha.view(-1, 1)
    
    # Binary Search for Tau*
    tau_min = torch.full((B,), tau_min_val)
    tau_max = torch.full((B,), tau_max_val)
    
    for _ in range(bs_iters):
        tau_mid = (tau_min + tau_max) / 2.0
        tau_mid_expanded = tau_mid.view(-1, 1)
        q_mid = temperature_transform(p_i, tau_mid_expanded)
        h_cur = entropy(q_mid)
        mask_too_soft = h_cur > h_target
        tau_max = torch.where(mask_too_soft, tau_mid, tau_max)
        tau_min = torch.where(~mask_too_soft, tau_mid, tau_min)
        
    tau_star = (tau_min + tau_max) / 2.0
    
    # Virtual Teacher & Distillation
    log_p_i = torch.log(p_i + 1e-10)
    z_t = log_p_i / tau_star.view(-1, 1)
    total_tau = tau_star.view(-1, 1) * tau_kd
    q_distill = temperature_transform(p_i, total_tau)
    
    print(f"{'Diff':<6} | {'H_tgt':<6} | {'Tau*':<6} | {'Top-5 Probs (q_distill)'}")
    print("-" * 60)
    
    for i in range(B):
        probs, _ = torch.sort(q_distill[i], descending=True)
        top5 = probs[:5].tolist()
        top5_str = ", ".join([f"{p:.4f}" for p in top5])
        print(f"{d_i[i]:.4f} | {h_target[i]:.4f} | {tau_star[i]:.4f} | [{top5_str}]")

if __name__ == "__main__":
    verify()
