import torch
import torch.nn as nn
import torch.nn.functional as F
from .distiller_base import Distiller


class PRISM2D(Distiller):
    def __init__(self, student, teacher, optimizer, device, cfg):
        super(PRISM2D, self).__init__(student, teacher, optimizer, device, cfg)
        
        # EMA decay rates
        self.var_ema_decay = cfg.distill.get('var_ema_decay', 0.99)
        self.dir_ema_decay = cfg.distill.get('dir_ema_decay', 0.9)
        
        # Online estimates
        self.sigma_h_sq_ema = None
        self.sigma_s_sq_ema = None
        self.u_parallel_ema = None
    
    def estimate_intra_batch_variance(self, g_samples):
        B = g_samples.size(0)
        if B < 2:
            return torch.tensor(1e-6, device=g_samples.device)
        
        g_mean = g_samples.mean(dim=0, keepdim=True)
        sample_var = ((g_samples - g_mean) ** 2).sum() / (B - 1)
        return sample_var / B
    
    def train_step(self, inputs, targets):
        self.student.train()
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        B = inputs.size(0)
        
        # Forward
        s_logits = self.student(inputs)
        with torch.no_grad():
            t_logits = self.teacher(inputs)
        
        T = self.cfg.distill.temperature
        T_sq = T * T
        C = s_logits.size(-1)
        
        # ============================================
        # Step 1: Compute Gradients on Parameters
        # ============================================
        
        # Hard Loss (CE)
        loss_ce = F.cross_entropy(s_logits, targets)
        
        # Soft Loss (KL) - Scaled by T^2
        loss_kl = nn.KLDivLoss(reduction="batchmean")(
            F.log_softmax(s_logits / T, dim=-1),
            F.softmax(t_logits / T, dim=-1)
        ) * T_sq
        
        # Compute Gradients
        params = list(self.student.parameters())
        
        # g_h (Parameter Gradient)
        grads_h = torch.autograd.grad(loss_ce, params, retain_graph=True)
        g_h_vec = torch.cat([g.view(-1) for g in grads_h])
        
        # g_s (Parameter Gradient, Scaled by T^2)
        grads_s = torch.autograd.grad(loss_kl, params, retain_graph=False)
        g_s_vec = torch.cat([g.view(-1) for g in grads_s])
        
        # g_s_unscaled (for weight calculation)
        g_s_unscaled_vec = g_s_vec / T_sq
        
        # ============================================
        # Step 2: Variance Estimation (Proxy via Logits)
        # ============================================
        # We use logit gradients to estimate the "noise level" scalar.
        with torch.no_grad():
            p_s = F.softmax(s_logits, dim=-1)
            y_onehot = F.one_hot(targets, num_classes=C).float()
            p_s_T = F.softmax(s_logits / T, dim=-1)
            p_t_T = F.softmax(t_logits / T, dim=-1)
            
            # Per-sample logit gradients (UNSCALED for g_s)
            g_h_logits_samples = p_s - y_onehot
            g_s_unscaled_logits_samples = p_s_T - p_t_T
            
            sigma_h_sq_batch = self.estimate_intra_batch_variance(g_h_logits_samples)
            sigma_s_sq_batch = self.estimate_intra_batch_variance(g_s_unscaled_logits_samples)
            
            if self.sigma_h_sq_ema is None:
                self.sigma_h_sq_ema = sigma_h_sq_batch.detach()
                self.sigma_s_sq_ema = sigma_s_sq_batch.detach()
            else:
                self.sigma_h_sq_ema = (self.var_ema_decay * self.sigma_h_sq_ema 
                                      + (1 - self.var_ema_decay) * sigma_h_sq_batch.detach())
                self.sigma_s_sq_ema = (self.var_ema_decay * self.sigma_s_sq_ema 
                                      + (1 - self.var_ema_decay) * sigma_s_sq_batch.detach())
            
            sigma_h_sq = self.sigma_h_sq_ema
            sigma_s_sq = self.sigma_s_sq_ema
        
        # ============================================
        # Step 3: EMA Direction
        # ============================================
        norm_g_h = torch.norm(g_h_vec)
        u_current = g_h_vec / (norm_g_h + 1e-8)
        
        if self.u_parallel_ema is None:
            self.u_parallel_ema = u_current.detach().clone()
        else:
            self.u_parallel_ema = (self.dir_ema_decay * self.u_parallel_ema 
                                   + (1 - self.dir_ema_decay) * u_current.detach())
            self.u_parallel_ema = self.u_parallel_ema / (torch.norm(self.u_parallel_ema) + 1e-8)
        
        u_hat = self.u_parallel_ema
        
        # ============================================
        # Step 4: Projections (UNSCALED g_s)
        # ============================================
        g_h_par = torch.dot(g_h_vec, u_hat)
        g_s_unscaled_par = torch.dot(g_s_unscaled_vec, u_hat)
        
        g_h_perp = g_h_vec - g_h_par * u_hat
        g_s_unscaled_perp = g_s_unscaled_vec - g_s_unscaled_par * u_hat
        
        # ============================================
        # Step 5: Bias & Weight Computation
        # ============================================
        d_par = g_s_unscaled_par - g_h_par
        d_perp = g_s_unscaled_perp - g_h_perp
        norm_d_perp_sq = torch.norm(d_perp) ** 2
        
        # Variance Scaling for Parameters
        # We use the scalar variance from logits as a proxy for "per-dimension" variance?
        # Or "total variance"?
        # Logit variance sigma_h_sq is "sum of variances over C classes".
        # Parameter vector has P elements.
        # If we assume noise is distributed, total param variance ~ (P/C) * sigma_h_sq.
        # But wait, dot product projections (g_h_par) are scalars.
        # Variance of g_h_par (scalar) is roughly E[|g_h|^2] / P? No.
        # Let's stick to the heuristic: use the scalar sigma estimates directly.
        # If they are too small/large, the weights will saturate.
        
        # Parallel (Scalar)
        sigma_h_par_sq = sigma_h_sq
        sigma_s_par_sq = sigma_s_sq
        
        # Perpendicular (Vector)
        sigma_h_perp_sq = sigma_h_sq
        sigma_s_perp_sq = sigma_s_sq
        
        b_par_sq = F.relu(d_par ** 2 - (sigma_h_par_sq + sigma_s_par_sq))
        b_perp_sq = F.relu(norm_d_perp_sq - (sigma_h_perp_sq + sigma_s_perp_sq))
        
        beta_par = b_par_sq / (sigma_h_par_sq + 1e-10)
        beta_perp = b_perp_sq / (sigma_h_perp_sq + 1e-10)
        
        r_par = sigma_s_par_sq / (sigma_h_par_sq + 1e-10)
        r_perp = sigma_s_perp_sq / (sigma_h_perp_sq + 1e-10)
        
        w_par = torch.sigmoid(-torch.log(beta_par + r_par + 1e-10))
        w_perp = torch.sigmoid(-torch.log(beta_perp + r_perp + 1e-10))
        
        # ============================================
        # Step 6: Final Mixing (Beta Transformation)
        # ============================================
        beta_star_par = w_par / (T_sq + w_par * (1 - T_sq))
        beta_star_perp = w_perp / (T_sq + w_perp * (1 - T_sq))
        
        # Mix using Scaled g_s (g_s_vec)
        # g_s_par_scaled = T^2 * g_s_unscaled_par
        # But we already have g_s_vec. We can project it?
        # g_s_par_scaled = dot(g_s_vec, u_hat)
        # Yes, because dot is linear.
        
        g_s_par_scaled = torch.dot(g_s_vec, u_hat)
        g_s_perp_scaled = g_s_vec - g_s_par_scaled * u_hat
        
        g_final_par = beta_star_par * g_s_par_scaled + (1 - beta_star_par) * g_h_par
        g_final_perp = beta_star_perp * g_s_perp_scaled + (1 - beta_star_perp) * g_h_perp
        
        g_final_vec = g_final_par * u_hat + g_final_perp
        
        # Assign gradients
        offset = 0
        for param in self.student.parameters():
            numel = param.numel()
            g_param = g_final_vec[offset:offset+numel].view_as(param)
            param.grad = g_param
            offset += numel
            
        self.optimizer.step()
        
        # Logging
        self.log_dict = {
            "w_par": w_par.item(),
            "w_perp": w_perp.item(),
            "beta_star_par": beta_star_par.item(),
            "beta_star_perp": beta_star_perp.item(),
            "beta_par": beta_par.item(),
            "sigma_h_sq": sigma_h_sq.item(),
            "g_h_norm": norm_g_h.item(),
        }
