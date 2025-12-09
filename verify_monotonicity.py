import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def entropy(p):
    return -torch.sum(p * torch.log(p + 1e-10), dim=1)

def temperature_transform(p, tau):
    log_p = torch.log(p + 1e-10)
    scaled_logits = log_p / tau
    return F.softmax(scaled_logits, dim=1)

def verify():
    # Config
    B = 100
    C = 100
    alpha = 0.1
    power_law_exp = 5.0
    device = 'cpu'
    
    # Generate p_i as in our code
    targets = torch.randint(0, C, (B,), device=device)
    p_i = torch.zeros(B, C, device=device)
    p_i.scatter_(1, targets.view(-1, 1), (1 - alpha))
    
    r = torch.rand(B, C, device=device).pow(power_law_exp)
    r.scatter_(1, targets.view(-1, 1), 0.0)
    r_sum = r.sum(dim=1, keepdim=True) + 1e-8
    r_norm = r / r_sum
    p_i += r_norm * alpha
    
    # Check monotonicity
    taus = torch.linspace(0.1, 10.0, 1000)
    entropies = []
    
    # Track one sample
    sample_idx = 0
    p_sample = p_i[sample_idx:sample_idx+1]
    
    print(f"Checking monotonicity for {B} samples...")
    
    non_monotonic_count = 0
    
    for i in range(B):
        p_s = p_i[i:i+1]
        prev_h = -1.0
        is_mono = True
        
        h_list = []
        for tau in taus:
            tau_tensor = torch.tensor([[tau]], device=device)
            q = temperature_transform(p_s, tau_tensor)
            h = entropy(q).item()
            h_list.append(h)
            
            if h < prev_h:
                is_mono = False
                # print(f"Sample {i}: Non-monotonic at tau={tau:.2f} (H={h:.4f} < Prev={prev_h:.4f})")
            prev_h = h
            
        if not is_mono:
            non_monotonic_count += 1
            
    print(f"Non-monotonic samples: {non_monotonic_count}/{B}")
    
    if non_monotonic_count == 0:
        print("SUCCESS: Entropy is strictly monotonic with respect to temperature.")
    else:
        print("FAILURE: Entropy is NOT monotonic.")

if __name__ == "__main__":
    verify()
