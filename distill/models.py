import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import get_activation

import torchvision

import timm
import torch.nn as nn
import torchvision
from huggingface_hub import hf_hub_download
import torch

class ResNetWrapper(nn.Module):
    def __init__(self, num_classes=10, pretrained=True, repo_id=None):
        super().__init__()
        if repo_id:
            # Load from Hugging Face manually to handle architecture mismatch (CIFAR-10 ResNet)
            # Standard ResNet18 has 7x7 conv and maxpool. CIFAR-10 ResNet usually has 3x3 conv and no maxpool.
            self.net = timm.create_model("resnet18", pretrained=False, num_classes=num_classes)
            
            # Modify for CIFAR-10
            self.net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.net.maxpool = nn.Identity() # Remove maxpool
            
            if pretrained:
                # Download and load weights
                try:
                    weights_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
                    state_dict = torch.load(weights_path, map_location="cpu")
                    
                    # Handle potential key mismatches (e.g. 'net.' prefix or different head names)
                    # The error showed 'conv1.weight' mismatch, so keys likely match standard ResNet.
                    # But let's be safe.
                    self.net.load_state_dict(state_dict, strict=False) # strict=False to be safe, but we want to ensure core matches
                except Exception as e:
                    print(f"Failed to load weights from {repo_id}: {e}")
                    # Fallback or error out? User requested this model.
                    raise e
        else:
            # Use torchvision ResNet18
            weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
            self.net = torchvision.models.resnet18(weights=weights)
            # Replace fc if num_classes != 1000
            if num_classes != 1000:
                self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)
            
    def forward(self, x, return_hidden=False):
        # Timm ResNet forward structure might differ slightly from torchvision
        # But for standard ResNet, it's usually compatible or we can use features_only=True
        # Let's assume standard structure for now, but handle potential naming diffs if needed.
        # Actually, timm models have a forward_features method.
        
        if hasattr(self.net, 'forward_features'):
            # Timm style
            features = self.net.forward_features(x)
            # features is usually (B, C, H, W) before pooling for some models, or pooled.
            # For ResNet in timm, forward_features returns unpooled features (B, 512, 1, 1) or (B, 512, H, W)
            # Let's check output shape or just use standard forward if we don't need intermediate
            
            # Re-implementing forward for flexibility (penultimate layer access)
            # Timm ResNet: conv1, bn1, act1, maxpool, layer1-4, global_pool, fc
            x = self.net.conv1(x)
            x = self.net.bn1(x)
            x = self.net.act1(x)
            x = self.net.maxpool(x)

            x = self.net.layer1(x)
            x = self.net.layer2(x)
            x = self.net.layer3(x)
            x = self.net.layer4(x)
            
            x = self.net.global_pool(x)
            h = torch.flatten(x, 1)
            out = self.net.fc(h)
            
            if return_hidden:
                return out, h
            return out
        else:
            # Torchvision style
            x = self.net.conv1(x)
            x = self.net.bn1(x)
            x = self.net.relu(x)
            x = self.net.maxpool(x)

            x = self.net.layer1(x)
            x = self.net.layer2(x)
            x = self.net.layer3(x)
            x = self.net.layer4(x)

            x = self.net.avgpool(x)
            h = torch.flatten(x, 1)
            out = self.net.fc(h)
            
            if return_hidden:
                return out, h
            return out

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(128 * 4 * 4, num_classes) # Assuming 32x32 input (CIFAR)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x, return_hidden=False):
        x = self.pool(self.relu(self.bn1(self.conv1(x)))) # 32 -> 16
        x = self.pool(self.relu(self.bn2(self.conv2(x)))) # 16 -> 8
        x = self.pool(self.relu(self.bn3(self.conv3(x)))) # 8 -> 4
        
        h = torch.flatten(x, 1) # 128 * 4 * 4 = 2048
        out = self.fc(h)
        
        if return_hidden:
            return out, h
        return out

class RandomFeatureAdapter(nn.Module):
    def __init__(self, input_dim, projection_dim, activation="relu", init_scale=1.0, init_method="gaussian", scales=[1.0], depth=1):
        super().__init__()
        self.input_dim = input_dim
        self.projection_dim = projection_dim
        self.activation = get_activation(activation)
        self.init_scale = init_scale
        self.init_method = init_method
        self.scales = scales
        self.depth = depth
        
        # Deep RFM: List of random matrices
        # For depth=1, it's just U1.
        # For depth=2, it's U1 -> Act -> U2 -> Act.
        # We store them in a ModuleList or just register buffers dynamically.
        
        self.layers = nn.ModuleList()
        
        # Input Normalization (Crucial for Random Features)
        self.input_norm = nn.LayerNorm(input_dim)
        
        # First layer: Input -> Proj
        # We use the existing logic for the first layer (Multi-Scale support)
        U1 = self._init_random_weights(input_dim, projection_dim, scales)
        self.register_buffer("U_0", U1)
        self.register_buffer("b_0", torch.rand(projection_dim) * 2 - 1)
        
        # Subsequent layers (if depth > 1)
        # Proj -> Proj (We keep dimension constant at projection_dim)
        for i in range(1, depth):
            # For intermediate layers, we usually use single scale (1.0) or simple initialization
            # to avoid exploding variance.
            if init_method == "orthogonal":
                 u = torch.nn.init.orthogonal_(torch.empty(projection_dim, projection_dim))
            else:
                 u = torch.randn(projection_dim, projection_dim)
            
            # Scale adjustment for deep networks? 1/sqrt(D) is standard.
            # Our init_scale handles this roughly.
            u = u * init_scale
            
            self.register_buffer(f"U_{i}", u)
            self.register_buffer(f"b_{i}", torch.rand(projection_dim) * 2 - 1)
        
        # Learnable linear readout V
        self.V = None 

    def _init_random_weights(self, in_dim, out_dim, scales):
        # Handle Multi-Scale for the FIRST layer
        num_scales = len(scales)
        chunk_size = out_dim // num_scales
        remainder = out_dim % num_scales
        
        U_list = []
        for i, scale in enumerate(scales):
            dim = chunk_size + (1 if i < remainder else 0)
            if dim == 0: continue
            
            if self.init_method == "gaussian":
                u = torch.randn(dim, in_dim)
            elif self.init_method == "orthogonal":
                if dim > in_dim:
                    rows = []
                    for _ in range(dim // in_dim + 1):
                        rows.append(torch.nn.init.orthogonal_(torch.empty(in_dim, in_dim)))
                    u = torch.cat(rows, dim=0)[:dim]
                else:
                    u = torch.nn.init.orthogonal_(torch.empty(dim, in_dim))
            else:
                raise ValueError(f"Unknown init_method: {self.init_method}")
            
            U_list.append(u * scale * self.init_scale)
            
        return torch.cat(U_list, dim=0) 

    def initialize_from_data(self, x):
        """
        Initialize bias b and scale U based on data statistics.
        Only applies to the FIRST layer for now to ensure good input distribution.
        """
        U = getattr(self, "U_0")
        b = getattr(self, "b_0")
        
        with torch.no_grad():
            projected = F.linear(x, U)
            mean = projected.mean(dim=0)
            b.data = -mean
        
    def forward(self, x):
        out = self.input_norm(x)
        for i in range(self.depth):
            U = getattr(self, f"U_{i}")
            b = getattr(self, f"b_{i}")
            out = F.linear(out, U, b)
            out = self.activation(out)
        
        features = out
        
        if self.V is not None:
            return F.linear(features, self.V)
        else:
            return features

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super().__init__()
        layers = []
        in_d = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_d, hidden_dim))
            layers.append(nn.ReLU())
            in_d = hidden_dim
        layers.append(nn.Linear(in_d, output_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x, return_hidden=False):
        if not return_hidden:
            return self.net(x)
        
        # For feature distillation, we might want the last hidden state
        h = x
        for layer in self.net[:-1]:
            h = layer(h)
        out = self.net[-1](h)
        return out, h
