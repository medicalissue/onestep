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
            
            # Soft Correctness (Alpha)
            alpha_per_sample = t_probs.gather(1, targets.view(-1, 1)).squeeze()
            alpha_batch = alpha_per_sample.mean().item()
        
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
        
        # 3. PRISM-E Logic (Vectorized with _foreach_)
        gamma_fn = self.cfg.distill.get('gamma_fn', 'tanh')
        
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

        # Flatten for dot/norm (View is cheap)
        g_s_flat = [g.view(-1) for g in g_s_list]
        g_h_flat = [g.view(-1) for g in g_h_list]
        
        # Compute Norms (Vectorized)
        norms_s = torch._foreach_norm(g_s_flat, 2)
        norms_h = torch._foreach_norm(g_h_flat, 2)
        
        # Compute Dot Products (Vectorized Mul + Sum)
        # _foreach_mul returns list of element-wise products
        prods = torch._foreach_mul(g_s_flat, g_h_flat)
        dots = [p.sum() for p in prods]
        
        # Stack to tensors for scalar math
        norms_s_t = torch.stack(norms_s)
        norms_h_t = torch.stack(norms_h)
        dots_t = torch.stack(dots)
        
        # Cosine Similarity
        # Avoid div by zero
        denoms = norms_s_t * norms_h_t
        mask = (norms_s_t > 1e-8) & (norms_h_t > 1e-8)
        cos_phi = torch.zeros_like(dots_t)
        cos_phi[mask] = dots_t[mask] / denoms[mask]
        
        # Gamma Calculation (Vectorized)
        if gamma_fn == 'linear':
            gamma = 1.0 + alpha_batch * cos_phi
        elif gamma_fn == 'exp':
            gamma = torch.exp(alpha_batch * cos_phi)
        else: # tanh
            gamma = 1.0 + alpha_batch * torch.tanh(cos_phi)
            
        gamma = torch.clamp(gamma, min=0.1)
        
        # Update Rule: g_final = alpha * g_s + gamma * g_h
        # We can use _foreach_ operations for this update
        # g_final = alpha * g_s + gamma * g_h
        
        # 1. Scale g_h by gamma
        # gamma is a tensor, we need a list of scalars for _foreach_mul
        gamma_list = gamma.tolist()
        g_h_scaled = torch._foreach_mul(g_h_list, gamma_list)
        
        # 2. Scale g_s by alpha
        g_s_scaled = torch._foreach_mul(g_s_list, alpha_batch)
        
        # 3. Add them
        g_final_list = torch._foreach_add(g_h_scaled, g_s_scaled)
        
        # 4. Assign back to param.grad
        for param, g_final in zip(params_with_grad, g_final_list):
            param.grad = g_final
            
        self.optimizer.step()
        
        avg_gamma = gamma.mean().item()
        avg_cos = cos_phi.mean().item()
        
        self.log_dict = {
            "loss": loss_ce.item(), 
            "alpha_soft": alpha_batch, 
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
        # targets: labels (same as input_ids for CausalLM usually)
        
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        
        with torch.no_grad():
            t_outputs = self.teacher(inputs)
            t_logits = t_outputs.logits
            t_probs = F.softmax(t_logits, dim=-1)
            
            # Entropy-based Alpha (Token-wise)
            # H(p) = -sum(p * log(p))
            entropy = -torch.sum(t_probs * torch.log(t_probs + 1e-8), dim=-1)
            vocab_size = t_logits.size(-1)
            max_entropy = math.log(vocab_size)
            normalized_entropy = entropy / max_entropy
            alpha_token = 1.0 - normalized_entropy # High confidence -> High Alpha
            alpha_batch = alpha_token.mean().item()
        
        s_outputs = self.student(inputs)
        s_logits = s_outputs.logits
        
        # 1. Soft Gradient (g_s) - Head Only
        # We only want gradients for the LM Head.
        # Assuming the head is named 'lm_head' (standard in GPT-2/Llama)
        
        lm_head = self.student.lm_head
        
        T = 2.0 # Lower T for LLMs usually
        loss_kl = nn.KLDivLoss(reduction="batchmean")(
            F.log_softmax(s_logits / T, dim=-1),
            F.softmax(t_logits / T, dim=-1)
        ) * (T * T)
        
        self.optimizer.zero_grad()
        loss_kl.backward(retain_graph=True)
        
        # Capture Soft Gradient of Head
        if lm_head.weight.grad is not None:
            g_s = lm_head.weight.grad.clone()
        else:
            # Should not happen if requires_grad is True
            g_s = torch.zeros_like(lm_head.weight)
            
        self.optimizer.zero_grad()
        
        # 2. Hard Gradient (g_h) - Head Only
        # Standard Causal LM Loss (Shifted inside model usually, but here we compute manually or use model output)
        # HuggingFace models compute loss automatically if labels are provided.
        # But we need explicit backward on logits for Head-Only control? 
        # Actually, we can just run backward on the loss.
        
        # Shift logits and labels for Causal LM loss
        shift_logits = s_logits[..., :-1, :].contiguous()
        shift_labels = targets[..., 1:].contiguous()
        loss_ce = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        loss_ce.backward()
        
        # Capture Hard Gradient of Head
        if lm_head.weight.grad is not None:
            g_h = lm_head.weight.grad
        else:
            g_h = torch.zeros_like(lm_head.weight)
            
        # 3. PRISM-E Logic (Head-Only, Token-wise approximation)
        # Ideally we want token-wise gamma, but gradients are aggregated over batch & sequence.
        # So we compute gamma based on the aggregated gradients of the Head.
        # This is "Head-wise" PRISM, which is a good approximation for "Token-wise" since the Head is the bottleneck.
        
        g_h_flat = g_h.view(-1)
        g_s_flat = g_s.view(-1)
        
        norm_s = torch.norm(g_s_flat)
        norm_h = torch.norm(g_h_flat)
        
        if norm_s > 1e-8 and norm_h > 1e-8:
            cos_phi = torch.dot(g_s_flat, g_h_flat) / (norm_s * norm_h)
        else:
            cos_phi = torch.tensor(0.0, device=self.device)
            
        # Gamma (Tanh)
        gamma = 1.0 + alpha_batch * torch.tanh(cos_phi)
        gamma = torch.clamp(gamma, min=0.1)
        
        # Update Head Gradient
        g_final = alpha_batch * g_s + gamma * g_h
        lm_head.weight.grad = g_final
        
        # For other layers (Backbone), they already have g_h from loss_ce.backward().
        # We leave them as is (Standard Hard Training for Backbone, PRISM for Head).
        # This implements "Head-Only PRISM".
        
        self.optimizer.step()
        
        self.log_dict = {
            "loss": loss_ce.item(), 
            "alpha": alpha_batch, 
            "gamma": gamma.item(), 
            "cos": cos_phi.item()
        }
        return s_logits
