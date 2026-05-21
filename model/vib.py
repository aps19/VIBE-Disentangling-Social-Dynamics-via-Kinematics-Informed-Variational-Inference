import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import StochasticDepth
import numpy as np

class CausalVIB(nn.Module):
    def __init__(self, input_dim=768, latent_dim=256):
        super().__init__()
        self.aff_encoder = nn.Sequential(nn.Linear(input_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, latent_dim * 2))
        self.env_encoder = nn.Sequential(nn.Linear(input_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, latent_dim * 2))
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu
    def forward(self, x):
        mu_aff, logvar_aff = self.aff_encoder(x).chunk(2, dim=-1)
        mu_env, logvar_env = self.env_encoder(x).chunk(2, dim=-1)
        z_aff = self.reparameterize(mu_aff, logvar_aff)
        z_env = self.reparameterize(mu_env, logvar_env)
        return {'z_aff': z_aff, 'mu_aff': mu_aff, 'logvar_aff': logvar_aff, 'z_env': z_env, 'mu_env': mu_env, 'logvar_env': logvar_env}
