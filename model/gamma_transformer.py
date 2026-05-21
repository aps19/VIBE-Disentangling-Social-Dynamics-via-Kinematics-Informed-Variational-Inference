import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import StochasticDepth
import numpy as np

class DecoupledAdaLN(nn.Module):
    def __init__(self, dim, style_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        
        # Predicts: Baseline Scale, Baseline Shift, and Sensitivity to Gamma
        self.style_proj = nn.Sequential(
            nn.Linear(style_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim * 3) 
        )
        nn.init.zeros_(self.style_proj[-1].weight)
        nn.init.zeros_(self.style_proj[-1].bias)

    def forward(self, x, gamma, global_style):
        # 1. Project Global Style
        style_params = self.style_proj(global_style) # [B, D*3]
        scale_base, shift_base, sync_sensitivity = style_params.chunk(3, dim=-1)
        
        # Expand for Time: [B, 1, D] -> [B, T, D]
        scale_base = scale_base.unsqueeze(1)
        shift_base = shift_base.unsqueeze(1)
        sync_sensitivity = sync_sensitivity.unsqueeze(1)
        
        # 2. Align Gamma Dimensions [B, T_gamma, 1] -> [B, T_x, 1]
        if gamma.dim() == 2: 
            gamma = gamma.unsqueeze(1) # [B, 1, 1]
            
        if gamma.shape[1] != x.shape[1]:
            if gamma.shape[1] == 1:
                # Static value expansion
                gamma = gamma.expand(-1, x.shape[1], -1)
            else:
                # Interpolation (e.g., 16 -> 32)
                # Permute to [B, Channels, Time] for interpolate
                gamma = gamma.permute(0, 2, 1) 
                gamma = F.interpolate(gamma, size=x.shape[1], mode='linear', align_corners=False)
                gamma = gamma.permute(0, 2, 1)

        # 3. Modulation: Baseline + (Intensity * Sensitivity)
        dynamic_scale = scale_base + (gamma * sync_sensitivity)
        
        return self.norm(x) * (1 + dynamic_scale) + shift_base

class GammaGatedTransformer(nn.Module):
    def __init__(self, dim, style_dim, depth=2, heads=8, mlp_dim=1024, dropout=0.1, drop_path=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.drop_path = StochasticDepth(p=drop_path, mode="batch")
        
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                DecoupledAdaLN(dim, style_dim),
                nn.MultiheadAttention(dim, heads, batch_first=True, dropout=dropout),
                DecoupledAdaLN(dim, style_dim),
                nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(mlp_dim, dim))
            ]))
            
    def forward(self, x, gamma, global_style):
        for norm1, attn, norm2, mlp in self.layers:
            x_norm = norm1(x, gamma, global_style)
            attn_out, _ = attn(x_norm, x_norm, x_norm)
            x = x + self.drop_path(attn_out)
            x_norm = norm2(x, gamma, global_style)
            x = x + self.drop_path(mlp(x_norm))
        return x
