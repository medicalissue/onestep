import torch
import torch.nn as nn

class Distiller(object):
    def __init__(self, student, teacher, optimizer, device, cfg):
        self.student = student
        self.teacher = teacher
        self.optimizer = optimizer
        self.device = device
        self.cfg = cfg
        self.log_dict = {}

    def train_step(self, inputs, targets):
        raise NotImplementedError
