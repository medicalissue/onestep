import torch
import numpy as np

def compute_statistics(residuals):
    """
    Compute mean and standard deviation of residuals for normalization.
    Args:
        residuals: Tensor of shape (N, D)
    Returns:
        mean: Tensor of shape (D,)
        std: Tensor of shape (D,)
    """
    mean = residuals.mean(dim=0)
    std = residuals.std(dim=0)
    return mean, std

def get_activation(name):
    if name == "relu":
        return torch.nn.ReLU()
    elif name == "gelu":
        return torch.nn.GELU()
    else:
        raise ValueError(f"Unknown activation: {name}")
