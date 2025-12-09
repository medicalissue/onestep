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
    h_min = 0.1
    h_max = 2.5
    alpha_min = 0.02
    alpha_max = 0.15
    tau_min_val = 0.5
    tau_max_val = 100.0
    power_law_exp = 3.0
    tau_kd = 4.0
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
    
    # Hybrid Secant/Bisection Method (Brent's Method approximation)
    tau_min = torch.full((B,), tau_min_val)
    tau_max = torch.full((B,), tau_max_val)
    tau = (tau_min + tau_max) / 2.0
    z = torch.log(p_i + 1e-10)
    
    for _ in range(bs_iters):
        tau_expanded = tau.view(-1, 1)
        logits_scaled = z / tau_expanded
        q = F.softmax(logits_scaled, dim=1)
        h_cur = entropy(q)
        diff = h_cur - h_target
        
        # Update Bracket
        mask_too_soft = diff > 0
        tau_max = torch.where(mask_too_soft, tau, tau_max)
        tau_min = torch.where(~mask_too_soft, tau, tau_min)
        
        # Gradient
        E_z = (q * z).sum(dim=1)
        E_z2 = (q * z.pow(2)).sum(dim=1)
        var_z = E_z2 - E_z.pow(2)
        dH_dtau = var_z / (tau.pow(3) + 1e-10)
        
        # Secant Step
        update = diff / (dH_dtau + 1e-10)
        tau_secant = tau - update
        
        # Check Validity
        buffer = (tau_max - tau_min) * 0.1
        is_secant_valid = (tau_secant > tau_min + buffer) & (tau_secant < tau_max - buffer)
        
        # Fallback
        tau_bisection = (tau_min + tau_max) / 2.0
        
        tau = torch.where(is_secant_valid, tau_secant, tau_bisection)
        
    tau_star = tau
    
    # Virtual Teacher & Distillation
    # Teacher Target: q = softmax(z_base / tau*) -> Matches H_target
    # Student: p = softmax(z_s / tau_kd)
    # We want p -> q
    
    log_p_i = torch.log(p_i + 1e-10)
    z_t = log_p_i / tau_star.view(-1, 1)
    # Teacher target: q_distill
    total_tau = tau_star.view(-1, 1) * tau_kd
    q_distill = temperature_transform(p_i, total_tau)
    
    print(f"{'Diff':<6} | {'H_tgt':<6} | {'Tau*':<6} | {'H_actual':<8} | {'Top-5 Probs (q_distill)'}")
    print("-" * 75)
    
    for i in range(B):
        probs = q_distill[i]
        top5_probs, _ = torch.topk(probs, 5)
        top5_list = top5_probs.tolist()
        h_act = entropy(probs.unsqueeze(0)).item()
        
        print(f"{d_i[i].item():.4f} | {h_target[i].item():.4f} | {tau_star[i].item():.4f} | {h_act:.4f}   | {['{:.4f}'.format(p) for p in top5_list]}")

if __name__ == "__main__":
    verify()
