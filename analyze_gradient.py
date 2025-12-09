import torch
import torch.nn.functional as F

def entropy(p):
    return -torch.sum(p * torch.log(p + 1e-10), dim=1)

def analyze():
    B = 1
    C = 100
    power_law_exp = 3.0
    alpha = 0.1
    
    # Generate p_i (Sharp distribution)
    p_i = torch.zeros(B, C)
    p_i[0, 0] = 1.0 - alpha
    
    r = torch.rand(B, C).pow(power_law_exp)
    r[0, 0] = 0.0
    r_norm = r / r.sum()
    p_i += r_norm * alpha
    
    z = torch.log(p_i + 1e-10)
    
    print(f"{'Tau':<10} | {'Entropy':<10} | {'Gradient (dH/dtau)':<20} | {'Update Step (if diff=0.1)'}")
    print("-" * 65)
    
    for tau_val in [0.5, 1.0, 1.35, 2.0, 5.0, 10.0, 20.0, 30.0]:
        tau = torch.tensor([tau_val])
        
        # Forward
        logits = z / tau
        q = F.softmax(logits, dim=1)
        h = entropy(q).item()
        
        # Gradient
        E_z = (q * z).sum()
        E_z2 = (q * z.pow(2)).sum()
        var_z = E_z2 - E_z.pow(2)
        grad = (var_z / tau.pow(3)).item()
        
        # Hypothetical Update Step for a target diff of 0.1
        # step = diff / grad
        step = 0.1 / (grad + 1e-10)
        
        print(f"{tau_val:<10.4f} | {h:<10.4f} | {grad:<20.6f} | {step:<10.4f}")

if __name__ == "__main__":
    analyze()
