import torch
import torch.nn.functional as F
import math

def entropy(probs):
    return -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

def temperature_transform(p, tau):
    return F.softmax(torch.log(p + 1e-10) / tau, dim=1)

def solve_tau(p, h_target):
    tau_min = 0.01
    tau_max = 50.0
    for _ in range(20):
        tau = (tau_min + tau_max) / 2
        q = temperature_transform(p, torch.tensor([tau]))
        h = entropy(q)
        if h > h_target:
            tau_max = tau # Too soft -> need lower tau (wait, H increases with T? Yes. High T -> Flat -> High H)
            # Standard softmax: T->inf implies Uniform. T->0 implies OneHot.
            # So High T = High H.
            # If h > target (Too flat), we need Lower T.
            tau_max = tau
        else:
            tau_min = tau
    return tau

def run_test():
    torch.manual_seed(42)
    C = 100
    H_target = 1.5
    
    # 1. Fixed Noise r (Dirichlet-like)
    # Let's say beta=0.01 (Sparse)
    r = torch.rand(1, C).pow(10) # Simulate sparse noise
    r[0, 0] = 0.0 # GT is index 0
    r = r / r.sum()
    
    # 2. Case A: Low Alpha (0.01) -> Base is very sharp
    alpha_1 = 0.01
    p1 = torch.zeros(1, C)
    p1[0, 0] = 1 - alpha_1
    p1 += alpha_1 * r
    
    # 3. Case B: High Alpha (0.20) -> Base is softer
    alpha_2 = 0.20
    p2 = torch.zeros(1, C)
    p2[0, 0] = 1 - alpha_2
    p2 += alpha_2 * r
    
    # 4. Solve for Tau
    tau1 = solve_tau(p1, H_target)
    tau2 = solve_tau(p2, H_target)
    
    q1 = temperature_transform(p1, torch.tensor([tau1]))
    q2 = temperature_transform(p2, torch.tensor([tau2]))
    
    print(f"Target H: {H_target}")
    
    print(f"\nCase A (Alpha={alpha_1}):")
    print(f"Tau: {tau1:.4f}")
    print(f"Top-5 q: {q1[0, :5].tolist()}")
    print(f"GT Prob: {q1[0, 0]:.4f}")
    
    print(f"\nCase B (Alpha={alpha_2}):")
    print(f"Tau: {tau2:.4f}")
    print(f"Top-5 q: {q2[0, :5].tolist()}")
    print(f"GT Prob: {q2[0, 0]:.4f}")
    
    diff = (q1 - q2).abs().sum().item()
    print(f"\nTotal L1 Difference between q1 and q2: {diff:.6f}")
    
    if diff < 1e-3:
        print("CONCLUSION: Alpha is REDUNDANT. Final q is identical.")
    else:
        print("CONCLUSION: Alpha MATTERS. Final q is different.")

if __name__ == "__main__":
    run_test()
