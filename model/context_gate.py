import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import StochasticDepth
import numpy as np

class ContextInjectionGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.context_net = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Dropout(0.1), nn.Linear(dim, dim * 2)
        )
        with torch.no_grad():
            self.context_net[-1].weight.zero_()
            self.context_net[-1].bias.zero_()
    def forward(self, z_aff, z_env):
        scale, shift = self.context_net(z_env).chunk(2, dim=-1)
        return self.norm(z_aff) * (1 + scale) + shift
