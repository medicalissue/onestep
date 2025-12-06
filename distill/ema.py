import torch
import torch.nn as nn
from copy import deepcopy

class ModelEMA:
    def __init__(self, model, decay=0.999, device=None):
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device:
            self.module.to(device)

    def update(self, model):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device:
                    model_v = model_v.to(self.device)
                ema_v.copy_(self.decay * ema_v + (1. - self.decay) * model_v)

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)
