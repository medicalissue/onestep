import torch
import torch.nn as nn
import torch.nn.functional as F
from .distiller_base import Distiller

class KD(Distiller):
    def __init__(self, student, teacher, optimizer, device, cfg):
        super(KD, self).__init__(student, teacher, optimizer, device, cfg)
        self.temperature = cfg.distill.temperature
        self.alpha = cfg.distill.alpha

    def train_step(self, inputs, targets):
        self.student.train()
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        
        # Forward
        s_logits = self.student(inputs)
        with torch.no_grad():
            t_logits = self.teacher(inputs)
            
        # Loss
        loss_ce = F.cross_entropy(s_logits, targets)
        
        loss_kl = nn.KLDivLoss(reduction="batchmean")(
            F.log_softmax(s_logits / self.temperature, dim=-1),
            F.softmax(t_logits / self.temperature, dim=-1)
        ) * (self.temperature ** 2)
        
        loss = (1 - self.alpha) * loss_ce + self.alpha * loss_kl
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Logging
        self.log_dict = {
            "loss_ce": loss_ce.item(),
            "loss_kl": loss_kl.item(),
            "loss_total": loss.item()
        }
