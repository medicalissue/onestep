import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from .distiller_base import Distiller

class PCGrad(Distiller):
    def __init__(self, student, teacher, optimizer, device, cfg):
        super(PCGrad, self).__init__(student, teacher, optimizer, device, cfg)
        self.temperature = cfg.distill.temperature
        self.alpha = cfg.distill.alpha

    def train_step(self, inputs, targets):
        self.student.train()
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        
        # Forward
        s_logits = self.student(inputs)
        with torch.no_grad():
            t_logits = self.teacher(inputs)
            
        # Define Tasks
        # Task 1: CE
        loss_ce = F.cross_entropy(s_logits, targets)
        
        # Task 2: KD
        loss_kl = nn.KLDivLoss(reduction="batchmean")(
            F.log_softmax(s_logits / self.temperature, dim=-1),
            F.softmax(t_logits / self.temperature, dim=-1)
        ) * (self.temperature ** 2)
        
        # Compute Gradients for each task
        self.optimizer.zero_grad()
        
        # We need to compute gradients separately.
        # Note: optimizer.zero_grad() clears grads.
        
        grads_ce = torch.autograd.grad(loss_ce, self.student.parameters(), retain_graph=True)
        grads_kl = torch.autograd.grad(loss_kl, self.student.parameters(), retain_graph=False)
        
        # Flatten gradients
        g_ce = torch.cat([g.view(-1) for g in grads_ce])
        g_kl = torch.cat([g.view(-1) for g in grads_kl])
        
        # PCGrad Logic
        # Check cosine similarity
        dot_prod = torch.dot(g_ce, g_kl)
        
        if dot_prod < 0:
            # Conflict! Project each onto the normal plane of the other?
            # PCGrad paper: "Project gradient g_i onto the normal plane of g_j"
            # g_i = g_i - (g_i . g_j) / ||g_j||^2 * g_j
            
            # Usually PCGrad is applied by sampling task order.
            # Here we have fixed 2 tasks.
            # Let's project both to be safe (symmetric).
            # Or prioritize CE?
            # Standard PCGrad:
            # For task i in random_shuffle(tasks):
            #   For task j in other_tasks:
            #     if g_i . g_j < 0:
            #       g_i = g_i - proj
            
            # Let's do randomized order
            tasks = [(g_ce, 'ce'), (g_kl, 'kl')]
            random.shuffle(tasks)
            
            g_1, name_1 = tasks[0]
            g_2, name_2 = tasks[1]
            
            # Project g_1 w.r.t g_2
            dot = torch.dot(g_1, g_2)
            if dot < 0:
                g_1 = g_1 - (dot / (torch.norm(g_2)**2 + 1e-8)) * g_2
                
            # Project g_2 w.r.t g_1 (modified g_1? No, original g_1 usually, but PCGrad is iterative)
            # "The projection is applied iteratively"
            # Since we only have 2 tasks, let's just project the second one against the (potentially projected) first one?
            # Actually, if we just want to remove conflict:
            # g_final = g_ce + g_kl (but modified)
            
            # Let's follow the iterative approach:
            # g_final = g_1_proj + g_2_proj? No.
            # We want a single update direction.
            # g_final = g_1 + g_2
            
            # Let's stick to the simple 2-task projection:
            # If conflict, remove the conflicting component from the *weaker* task?
            # Or just project both?
            
            # Implementation from widely used PCGrad repos:
            # grads = [g_ce, g_kl]
            # pc_grads = copy(grads)
            # for i in range(2):
            #   for j in range(2):
            #     if i == j: continue
            #     g_i = pc_grads[i]
            #     g_j = grads[j] # Original grad
            #     dot = g_i.dot(g_j)
            #     if dot < 0:
            #       pc_grads[i] -= (dot / g_j.norm()**2) * g_j
            
            g_ce_pc = g_ce.clone()
            g_kl_pc = g_kl.clone()
            
            # Project CE w.r.t KL
            if dot_prod < 0:
                g_ce_pc = g_ce - (dot_prod / (torch.norm(g_kl)**2 + 1e-8)) * g_kl
            
            # Project KL w.r.t CE
            if dot_prod < 0:
                g_kl_pc = g_kl - (dot_prod / (torch.norm(g_ce)**2 + 1e-8)) * g_ce
                
            # Combine
            # We apply alpha weighting here?
            # PCGrad usually assumes sum of losses.
            # Our loss is (1-alpha)CE + alpha*KL.
            # So we should project the *weighted* gradients.
            
            # Re-compute weighted gradients
            g_ce_w = (1 - self.alpha) * g_ce
            g_kl_w = self.alpha * g_kl
            
            dot_w = torch.dot(g_ce_w, g_kl_w)
            
            g_ce_final = g_ce_w
            g_kl_final = g_kl_w
            
            if dot_w < 0:
                g_ce_final = g_ce_w - (dot_w / (torch.norm(g_kl_w)**2 + 1e-8)) * g_kl_w
                g_kl_final = g_kl_w - (dot_w / (torch.norm(g_ce_w)**2 + 1e-8)) * g_ce_w
                
            g_final = g_ce_final + g_kl_final
            conflict = 1.0
            
        else:
            # No conflict
            g_final = (1 - self.alpha) * g_ce + self.alpha * g_kl
            conflict = 0.0
            
        # Assign gradients
        offset = 0
        for param in self.student.parameters():
            numel = param.numel()
            g_param = g_final[offset:offset+numel].view_as(param)
            param.grad = g_param
            offset += numel
            
        self.optimizer.step()
        
        # Logging
        self.log_dict = {
            "loss_ce": loss_ce.item(),
            "loss_kl": loss_kl.item(),
            "conflict": conflict,
            "cos_sim": dot_prod.item() / (torch.norm(g_ce)*torch.norm(g_kl) + 1e-8)
        }
