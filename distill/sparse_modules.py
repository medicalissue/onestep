import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAutoEncoder(nn.Module):
    def __init__(self, hidden_dim, expansion_factor=4, sparsity_lambda=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = int(hidden_dim * expansion_factor)
        self.sparsity_lambda = sparsity_lambda
        
        # Encoder & Decoder
        self.encoder = nn.Linear(hidden_dim, self.latent_dim)
        self.decoder = nn.Linear(self.latent_dim, hidden_dim)
        
        # Initialize decoder with transpose of encoder (optional, but good for stability)
        # self.decoder.weight.data = self.encoder.weight.data.t()
        
    def shrinkage(self, x):
        # Soft-thresholding for L1 sparsity
        # S_lambda(x) = sign(x) * max(|x| - lambda, 0)
        # But for differentiability, ReLU-based approach is often used or just L1 penalty in loss
        # Here we implement ReLU as a simple non-linearity that induces sparsity naturally
        # Or we can use a custom soft-thresholding function
        return F.relu(x) 

    def forward(self, x):
        # x: [Batch, Seq, Hidden]
        
        # Encode
        u = self.encoder(x) # [B, S, K]
        
        # Shrinkage / Activation
        z = self.shrinkage(u) # [B, S, K]
        
        # Decode
        x_hat = self.decoder(z) # [B, S, H]
        
        return x_hat, z, u
