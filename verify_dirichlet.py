import torch
import matplotlib.pyplot as plt

def analyze_dirichlet(beta, C=100, samples=5):
    print(f"\n--- Analyzing Dirichlet(beta={beta}, C={C}) ---")
    m = torch.distributions.Dirichlet(torch.full((samples, C), beta))
    r = m.sample()
    
    # Check sparsity (values close to 0)
    threshold = 1e-4
    near_zeros = (r < threshold).float().mean().item() * 100
    
    print(f"Avg % of values < {threshold}: {near_zeros:.2f}%")
    
    # Show Top-10 of first sample
    top_vals, _ = torch.topk(r[0], 10)
    print(f"Top 10 values (Sample 0): {top_vals.tolist()}")
    
    # Show Bottom-10 of first sample
    bottom_vals, _ = torch.topk(r[0], 10, largest=False)
    print(f"Bottom 10 values (Sample 0): {bottom_vals.tolist()}")

if __name__ == "__main__":
    analyze_dirichlet(0.01)
    analyze_dirichlet(0.1)
    analyze_dirichlet(1.0)
