import torch
import logging

log = logging.getLogger(__name__)

class OnePassDistiller:
    def __init__(self, feature_dim, output_dim, lambda_reg=1e-4, device="cpu"):
        """
        Args:
            feature_dim: Dimension of the random features (D)
            output_dim: Dimension of the target residuals (d_out)
            lambda_reg: Ridge regression regularization coefficient
            device: torch device
        """
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.lambda_reg = lambda_reg
        self.device = device
        
        # Initialize statistics
        # A = Phi^T * Phi + lambda * I
        # Use float64 for stability
        self.A = torch.eye(feature_dim, device=device, dtype=torch.float64) * lambda_reg
        # B = Phi^T * r
        self.B = torch.zeros(feature_dim, output_dim, device=device, dtype=torch.float64)
        
        self.num_samples = 0

    def accumulate_batch(self, features, residuals):
        """
        Accumulate statistics from a batch.
        Args:
            features: (Batch, D) - The random features Phi(x)
            residuals: (Batch, d_out) - The target residuals r(x)
        """
        batch_size = features.shape[0]
        self.num_samples += batch_size
        
        # Move to device if needed and cast to float64
        features = features.to(self.device).double()
        residuals = residuals.to(self.device).double()
        
        # Update A: A += Phi^T * Phi
        # (D, B) @ (B, D) -> (D, D)
        self.A += torch.matmul(features.T, features)
        
        # Update B: B += Phi^T * r
        # (D, B) @ (B, d_out) -> (D, d_out)
        self.B += torch.matmul(features.T, residuals)

    def solve(self):
        """
        Solve for W* = A^{-1} B
        Returns:
            W: (D, d_out)
        """
        log.info(f"Solving linear system with D={self.feature_dim} on {self.device}...")
        
        # CPU Offloading for large D
        # If D is very large (e.g. > 20k), GPU memory might be insufficient for Cholesky.
        # We can move A and B to CPU, solve there, and move back.
        original_device = self.device
        if self.feature_dim > 20000 and self.device.type == 'cuda':
            log.info("Dimension > 20k, offloading solve to CPU to avoid OOM...")
            solve_device = torch.device('cpu')
        else:
            solve_device = self.device
            
        A = self.A.to(solve_device)
        B = self.B.to(solve_device)
        
        try:
            # Cholesky solve
            # W = A^{-1} B
            # L = cholesky(A)
            # W = cholesky_solve(B, L)
            L = torch.linalg.cholesky(A)
            W = torch.cholesky_solve(B, L)
        except RuntimeError as e:
            log.warning(f"Cholesky solve failed: {e}. Falling back to lstsq (slower but more robust).")
            W = torch.linalg.lstsq(A, B).solution
            
        return W.T.to(original_device).float() # Return as (d_out, D)
