import torch
import torch.nn as nn
import torch.nn.functional as F
from .distiller_base import Distiller

class DKD(Distiller):
    def __init__(self, student, teacher, optimizer, device, cfg):
        super(DKD, self).__init__(student, teacher, optimizer, device, cfg)
        self.temperature = cfg.distill.temperature
        self.alpha = cfg.distill.get('dkd_alpha', 1.0)
        self.beta = cfg.distill.get('dkd_beta', 8.0)

    def dkd_loss(self, logits_student, logits_teacher, target, alpha, beta, temperature):
        gt_mask = _get_gt_mask(logits_student, target)
        other_mask = _get_other_mask(logits_student, target)
        
        pred_student = F.softmax(logits_student / temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
        
        pred_student = cat_mask(pred_student, gt_mask, other_mask)
        pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
        
        log_pred_student = torch.log(pred_student)
        
        tckd_loss = (
            F.kl_div(log_pred_student, pred_teacher, size_average=False)
            * (temperature**2)
            / target.shape[0]
        )
        
        pred_teacher_part2 = F.softmax(
            logits_teacher / temperature - 1000.0 * gt_mask, dim=1
        )
        log_pred_student_part2 = F.log_softmax(
            logits_student / temperature - 1000.0 * gt_mask, dim=1
        )
        
        nckd_loss = (
            F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
            * (temperature**2)
            / target.shape[0]
        )
        
        return alpha * tckd_loss + beta * nckd_loss

    def train_step(self, inputs, targets):
        self.student.train()
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        
        # Forward
        s_logits = self.student(inputs)
        with torch.no_grad():
            t_logits = self.teacher(inputs)
            
        # Loss
        loss_ce = F.cross_entropy(s_logits, targets)
        loss_dkd = self.dkd_loss(s_logits, t_logits, targets, self.alpha, self.beta, self.temperature)
        
        loss = loss_ce + loss_dkd
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Logging
        self.log_dict = {
            "loss_ce": loss_ce.item(),
            "loss_dkd": loss_dkd.item(),
            "loss_total": loss.item()
        }

def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask

def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask

def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt
