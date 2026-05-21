import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import StochasticDepth
import numpy as np

# ==========================================
# 1. ATTENTION POOLING
# ==========================================
class LearnedQueryAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.attn_proj = nn.Linear(input_dim, hidden_dim)
        self.query_vector = nn.Parameter(torch.randn(hidden_dim, 1))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        u = torch.tanh(self.attn_proj(x)) 
        scores = torch.matmul(u, self.query_vector).squeeze(-1) 
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=1) 
        context_vector = torch.sum(x * attn_weights.unsqueeze(-1), dim=1) 
        return context_vector, attn_weights


class CrossModalAttentionBlock(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=dropout)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim))
        self.dropout = nn.Dropout(dropout)
    def forward(self, vis, aud):
        kv = aud
        q = vis
        attn_out, _ = self.cross_attn(q, kv, kv)
        vis = self.norm1(vis + self.dropout(attn_out))
        attn_out, _ = self.self_attn(vis, vis, vis)
        vis = self.norm2(vis + self.dropout(attn_out))
        vis = self.norm3(vis + self.ffn(vis))
        return vis
