import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BaseDistiller:
    def __init__(self, student, teacher, optimizer, device, cfg):
        self.student = student
        self.teacher = teacher
        self.optimizer = optimizer
        self.device = device
        self.cfg = cfg
        self.log_dict = {}

    def train_step(self, inputs, targets):
        raise NotImplementedError

class KD(BaseDistiller):
    def train_step(self, inputs, targets):
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        
        with torch.no_grad():
            t_logits = self.teacher(inputs)
        
        s_logits = self.student(inputs)
        
        # Soft Loss (KL)
        T = self.cfg.distill.temperature
        alpha = self.cfg.distill.get('alpha', 1.0)
        
        loss_kl = nn.KLDivLoss(reduction="batchmean")(
            F.log_softmax(s_logits / T, dim=1),
            F.softmax(t_logits / T, dim=1)
        ) * (T * T)
        
        # Hard Loss (CE)
        loss_ce = F.cross_entropy(s_logits, targets)
        
        # Final Loss (Standard KD: alpha * Soft + (1-alpha) * Hard)
        # But in our framework, we usually treat them as separate terms.
        # Let's stick to the user's implicit formulation: Loss = Soft + Hard (or weighted)
        # In previous distill.py, it was just loss_kl.backward() then loss_ce.backward().
        # This implies Loss = Soft + Hard.
        
        loss = loss_kl + loss_ce
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.log_dict = {"loss": loss.item(), "loss_kl": loss_kl.item(), "loss_ce": loss_ce.item()}
        return s_logits

class PRISM(BaseDistiller):
    def train_step(self, inputs, targets):
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        
        with torch.no_grad():
            t_logits = self.teacher(inputs)
            t_probs = F.softmax(t_logits, dim=1)
            
            # --- 1. Relative Correctness Score (S_rel) ---
            # S_rel = P_T(y_GT) / max(P_T)
            # If teacher predicts GT as #1, S_rel = 1.0
            # If teacher predicts GT as #2 (near-miss), S_rel ~ 0.8-0.9
            # If teacher hallucinates, S_rel ~ 0.0
            
            prob_gt = t_probs.gather(1, targets.view(-1, 1)).squeeze()
            prob_max = t_probs.max(dim=1).values
            
            # Avoid div by zero (though prob_max should be > 0)
            s_rel_per_sample = prob_gt / (prob_max + 1e-8)
            s_rel_batch = s_rel_per_sample.mean().item()
            
            # --- 2. Entropy-Aware Confidence (Alpha) ---
            entropy = -torch.sum(t_probs * F.log_softmax(t_logits, dim=1), dim=1)
            max_entropy = math.log(t_probs.size(1))
            alpha_per_sample = 1.0 - (entropy / max_entropy)
            alpha_batch = alpha_per_sample.mean().item()
            alpha_batch = max(0.0, min(1.0, alpha_batch))
        
        s_logits = self.student(inputs)
        
        # 1. Soft Gradient (g_s)
        T = self.cfg.distill.temperature
        loss_kl = nn.KLDivLoss(reduction="batchmean")(
            F.log_softmax(s_logits / T, dim=1),
            F.softmax(t_logits / T, dim=1)
        ) * (T * T)
        
        self.optimizer.zero_grad()
        loss_kl.backward(retain_graph=True)
        
        # Store Soft Gradients
        grad_s = {}
        for name, param in self.student.named_parameters():
            if param.grad is not None:
                grad_s[name] = param.grad.clone()
        
        self.optimizer.zero_grad()
        
        # 2. Hard Gradient (g_h)
        loss_ce = F.cross_entropy(s_logits, targets)
        loss_ce.backward()
        
        # 3. PRISM Logic: Relative Geometric Gating
        # g_final = g_s + gamma * g_h
        # gamma = 1 + S_rel * alpha * tanh(cos)
        
        # Collect gradients
        params_with_grad = []
        g_s_list = []
        g_h_list = []
        
        for name, param in self.student.named_parameters():
            if param.grad is not None and name in grad_s:
                params_with_grad.append(param)
                g_s_list.append(grad_s[name])
                g_h_list.append(param.grad)
        
        if not g_s_list:
            self.optimizer.step()
            return s_logits

        # Flatten for dot/norm
        g_s_flat = [g.view(-1) for g in g_s_list]
        g_h_flat = [g.view(-1) for g in g_h_list]
        
        # Compute Norms
        norms_s = torch._foreach_norm(g_s_flat, 2)
        norms_h = torch._foreach_norm(g_h_flat, 2)
        
        # Compute Dot Products
        prods = torch._foreach_mul(g_s_flat, g_h_flat)
        dots = [p.sum() for p in prods]
        
        # Stack to tensors
        norms_s_t = torch.stack(norms_s)
        norms_h_t = torch.stack(norms_h)
        dots_t = torch.stack(dots)
        
        # Cosine Similarity
        denoms = norms_s_t * norms_h_t
        mask = (norms_s_t > 1e-8) & (norms_h_t > 1e-8)
        cos_phi = torch.zeros_like(dots_t)
        cos_phi[mask] = dots_t[mask] / denoms[mask]
        
        # --- Gamma Calculation (Relative Geometric Gating) ---
        # gamma = 1 + S_rel * alpha * tanh(cos)
        
        tanh_cos = torch.tanh(cos_phi)
        
        # Scalar factor for the batch
        factor = s_rel_batch * alpha_batch
        
        gamma = 1.0 + factor * tanh_cos
        
        # Update Rule: g_final = g_s + gamma * g_h
        
        # 1. Scale g_h by gamma
        gamma_list = gamma.tolist()
        g_h_scaled = torch._foreach_mul(g_h_list, gamma_list)
        
        # 2. Add g_s
        g_final_list = torch._foreach_add(g_h_scaled, g_s_list)
        
        # 3. Assign back
        for param, g_final in zip(params_with_grad, g_final_list):
            param.grad = g_final
            
        self.optimizer.step()
        
        avg_gamma = gamma.mean().item()
        avg_cos = cos_phi.mean().item()
        
        self.log_dict = {
            "loss": loss_ce.item(), 
            "alpha": alpha_batch,
            "s_rel": s_rel_batch,
            "gamma": avg_gamma,
            "cos": avg_cos
        }
        return s_logits

class PCGrad(BaseDistiller):
    def train_step(self, inputs, targets):
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        
        with torch.no_grad():
            t_logits = self.teacher(inputs)
        
        s_logits = self.student(inputs)
        
        # Soft Loss
        T = 4.0
        loss_kl = nn.KLDivLoss(reduction="batchmean")(
            F.log_softmax(s_logits / T, dim=1),
            F.softmax(t_logits / T, dim=1)
        ) * (T * T)
        
        self.optimizer.zero_grad()
        loss_kl.backward(retain_graph=True)
        
        grad_s = {}
        for name, param in self.student.named_parameters():
            if param.grad is not None:
                grad_s[name] = param.grad.clone()
        
        self.optimizer.zero_grad()
        
        # Hard Loss
        loss_ce = F.cross_entropy(s_logits, targets)
        loss_ce.backward()
        
        # PCGrad Logic
        for name, param in self.student.named_parameters():
            if param.grad is not None and name in grad_s:
                g_h = param.grad
                g_s = grad_s[name]
                
                g_h_flat = g_h.view(-1)
                g_s_flat = g_s.view(-1)
                
                dot = torch.dot(g_s_flat, g_h_flat)
                
                if dot < 0:
                    # Conflict: Project g_h onto normal plane of g_s
                    # g_h_proj = g_h - (dot / norm_s^2) * g_s
                    norm_s_sq = torch.dot(g_s_flat, g_s_flat)
                    if norm_s_sq > 1e-8:
                        g_h = g_h - (dot / norm_s_sq) * g_s
                
                # Final Gradient: g_s + g_h (Projected)
                param.grad = g_s + g_h
        
        self.optimizer.step()
        self.log_dict = {"loss": loss_ce.item()}
        return s_logits

class DKD(BaseDistiller):
    def train_step(self, inputs, targets):
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        
        with torch.no_grad():
            t_logits = self.teacher(inputs)
        
        s_logits = self.student(inputs)
        
        # DKD Implementation
        # 1. TCKD (Target Class Knowledge Distillation)
        # 2. NCKD (Non-Target Class Knowledge Distillation)
        
        alpha = 1.0
        beta = 1.0 # Default DKD params
        T = 4.0
        
        gt_mask = torch.zeros_like(t_logits).scatter_(1, targets.view(-1, 1), 1).bool()
        other_mask = ~gt_mask
        
        t_probs = F.softmax(t_logits / T, dim=1)
        s_probs = F.softmax(s_logits / T, dim=1)
        
        t_pt = (t_probs * gt_mask).sum(dim=1, keepdim=True)
        s_pt = (s_probs * gt_mask).sum(dim=1, keepdim=True)
        
        loss_tckd = F.kl_div(torch.log(s_pt + 1e-8), t_pt + 1e-8, reduction='batchmean')
        
        t_nt = t_probs[other_mask].view(t_probs.size(0), -1)
        s_nt = s_probs[other_mask].view(s_probs.size(0), -1)
        
        t_nt = t_nt / t_nt.sum(dim=1, keepdim=True)
        s_nt = s_nt / s_nt.sum(dim=1, keepdim=True)
        
        loss_nckd = F.kl_div(torch.log(s_nt + 1e-8), t_nt + 1e-8, reduction='batchmean')
        
        loss_ce = F.cross_entropy(s_logits, targets)
        
        loss = loss_ce + alpha * loss_tckd + beta * loss_nckd
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.log_dict = {"loss": loss.item(), "loss_tckd": loss_tckd.item(), "loss_nckd": loss_nckd.item()}
        return s_logits

class PRISM_LLM(BaseDistiller):
    def train_step(self, inputs, targets):
        # inputs: input_ids
        # targets: labels (same as input_ids for CausalLM usually, with -100 for padding)
        
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        
        with torch.no_grad():
            t_outputs = self.teacher(inputs)
            t_logits = t_outputs.logits
            t_probs = F.softmax(t_logits, dim=-1)
            
            # Shift targets for Causal LM (next token prediction)
            shift_probs = t_probs[..., :-1, :].contiguous()
            shift_targets = targets[..., 1:].contiguous()
            
            # Mask for valid tokens (ignore padding -100)
            valid_mask = (shift_targets != -100) # [B, L-1]
            
            # --- 1. Relative Correctness Score (S_rel) ---
            # Safe Gather: Replace -100 with 0 temporarily
            safe_targets = shift_targets.clone()
            safe_targets[~valid_mask] = 0
            
            # Gather prob of GT
            prob_gt = shift_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1) # [B, L-1]
            prob_gt = prob_gt * valid_mask.float() # Zero out invalid
            
            prob_max = shift_probs.max(dim=-1).values # [B, L-1]
            
            s_rel = prob_gt / (prob_max + 1e-8)
            
            # Mean over valid tokens only
            num_valid = valid_mask.sum().item()
            s_rel_batch = s_rel.sum().item() / max(1.0, num_valid)
            
            # --- 2. Entropy-Aware Confidence (Alpha) ---
            entropy = -torch.sum(shift_probs * torch.log(shift_probs + 1e-8), dim=-1)
            vocab_size = t_logits.size(-1)
            max_entropy = math.log(vocab_size)
            alpha_token = 1.0 - (entropy / max_entropy)
            alpha_token = torch.clamp(alpha_token, 0.0, 1.0)
            
            # Mean over valid tokens
            alpha_batch = (alpha_token * valid_mask.float()).sum().item() / max(1.0, num_valid)
            
            # Teacher Accuracy
            t_preds = shift_probs.argmax(dim=-1)
            correct = (t_preds == shift_targets) & valid_mask
            teacher_acc = correct.float().sum().item() / max(1.0, num_valid)
        
        # Enable hidden states for Semantic PRISM
        s_outputs = self.student(inputs, output_hidden_states=True)
        s_logits = s_outputs.logits
        
        # Hidden States: Tuple of (Embeddings, Layer 1, ..., Layer N)
        # We use transformer layers (1 to N)
        all_hidden_states = s_outputs.hidden_states[1:] 
        
        shift_s_logits = s_logits[..., :-1, :].contiguous()
        
        # --- Semantic Gradient Calculation (Layer-wise) ---
        
        # 1. Soft Loss (KL) - Unreduced
        T = 2.0
        loss_kl_elementwise = nn.KLDivLoss(reduction="none")(
            F.log_softmax(shift_s_logits / T, dim=-1),
            F.softmax(shift_probs / T, dim=-1)
        )
        loss_kl_token = loss_kl_elementwise.sum(dim=-1) # [B, L-1]
        
        # 2. Hard Loss (CE) - Unreduced
        loss_ce_token = F.cross_entropy(
            shift_s_logits.view(-1, vocab_size), 
            shift_targets.view(-1), 
            reduction='none'
        ).view(shift_targets.shape) # [B, L-1]
        
        # Calculate Gradients w.r.t ALL Hidden States
        # We sum the losses to get a scalar for autograd
        
        # Soft Gradients (Layer-wise)
        # retain_graph=True is needed
        grads_s_hidden = torch.autograd.grad(
            loss_kl_token.sum(), 
            all_hidden_states, 
            retain_graph=True
        )
        
        # Hard Gradients (Layer-wise)
        grads_h_hidden = torch.autograd.grad(
            loss_ce_token.sum(), 
            all_hidden_states, 
            retain_graph=True
        )
        
        # --- Layer-wise Gradient Mixing ---
        
        mixed_grads = []
        prism_lambda = self.cfg.distill.get('prism_lambda', 5.0)
        
        # Evidence Strength is common across layers (based on final output probabilities)
        evidence_strength = s_rel * alpha_token # [B, L-1]
        
        avg_w_prism_list = []
        avg_cos_list = []
        
        for i, (g_s, g_h) in enumerate(zip(grads_s_hidden, grads_h_hidden)):
            # g_s, g_h: [B, L, H]
            
            # Slice for semantic calculation (ignore last token)
            g_s_sem = g_s[:, :-1, :] # [B, L-1, H]
            g_h_sem = g_h[:, :-1, :] # [B, L-1, H]
            
            # Flatten for dot product
            g_s_flat = g_s_sem.reshape(-1, g_s_sem.size(-1)) # [N, H]
            g_h_flat = g_h_sem.reshape(-1, g_h_sem.size(-1)) # [N, H]
            
            # Dot product
            dot = torch.sum(g_s_flat * g_h_flat, dim=-1) # [N]
            
            # Norms
            norm_s = torch.norm(g_s_flat, dim=-1)
            norm_h = torch.norm(g_h_flat, dim=-1)
            
            # Cosine
            denom = norm_s * norm_h
            cos_flat = torch.zeros_like(dot)
            mask_cos = denom > 1e-8
            cos_flat[mask_cos] = dot[mask_cos] / denom[mask_cos]
            
            # Reshape back to [B, L-1]
            cos_layer = cos_flat.view(g_s_sem.shape[:2]) # [B, L-1]
            
            # --- Sigmoidal Gating (Layer-specific) ---
            # w_prism_l = Sigmoid(lambda * S_rel * alpha * cos_l)
            
            gating_input = prism_lambda * evidence_strength * cos_layer
            w_prism_layer = torch.sigmoid(gating_input) # [B, L-1]
            
            # --- Gradient Mixing ---
            # g_mix = w * g_s + (1-w) * g_h
            # We need to broadcast w to [B, L-1, H]
            w_expanded = w_prism_layer.unsqueeze(-1) # [B, L-1, 1]
            
            # Mix gradients for valid tokens
            g_mix_sem = w_expanded * g_s_sem + (1.0 - w_expanded) * g_h_sem
            
            # For the last token (L-th), the gradient should be zero because the loss 
            # is calculated on shifted logits ([:-1]).
            # We explicitly pad with zeros to match the shape [B, L, H].
            zeros_last = torch.zeros_like(g_s[:, -1:, :]) # [B, 1, H]
            
            # Concatenate back to [B, L, H]
            g_mix = torch.cat([g_mix_sem, zeros_last], dim=1)
            
            mixed_grads.append(g_mix)
            
            # Logging stats
            avg_w_prism_list.append((w_prism_layer * valid_mask.float()).sum().item() / max(1.0, num_valid))
            avg_cos_list.append((cos_layer * valid_mask.float()).sum().item() / max(1.0, num_valid))
            
        # --- Backward Mixed Gradients ---
        # We backward from hidden states to parameters
        self.optimizer.zero_grad()
        torch.autograd.backward(all_hidden_states, grad_tensors=mixed_grads)
        self.optimizer.step()
        
        # Logging
        # Average across layers
        avg_w_prism = sum(avg_w_prism_list) / len(avg_w_prism_list)
        avg_cos_final = sum(avg_cos_list) / len(avg_cos_list)
        avg_p_t_gt = prob_gt.sum().item() / max(1.0, num_valid)
        
        self.log_dict = {
            "loss": loss_ce_token.mean().item(), # Log CE loss as reference
            "alpha": alpha_batch, 
            "s_rel": s_rel_batch,
            "teacher_acc": teacher_acc,
            "avg_p_t_gt": avg_p_t_gt,
            "w_prism": avg_w_prism, 
            "semantic_cos_avg": avg_cos_final,
            "w_prism_L0": avg_w_prism_list[0] if avg_w_prism_list else 0,
            "w_prism_Llast": avg_w_prism_list[-1] if avg_w_prism_list else 0,
        }
        return s_logits
