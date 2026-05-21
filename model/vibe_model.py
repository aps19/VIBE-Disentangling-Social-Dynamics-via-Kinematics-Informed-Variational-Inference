import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import StochasticDepth
import numpy as np

from vib import CausalVIB
from context_gate import ContextInjectionGate
from learned_attenttion import LearnedQueryAttention, CrossModalAttentionBlock
from gamma_transformer import DecoupledAdaLN, GammaGatedTransformer


class VIBE_Transformer(nn.Module):
    def __init__(self, input_dim=768, latent_dim=256, num_classes=3):
        super().__init__()
        # --- Stage 1: Global Pre-processing ---
        self.spatial_pool = LearnedQueryAttention(input_dim=input_dim, hidden_dim=128)
        
        # Style Projector
        self.style_dim = 64
        self.global_style_proj = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, self.style_dim) 
        )
        self.scene_scaler = nn.Sequential(nn.Linear(self.style_dim, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())

        # --- Stage 2: Disentanglement ---
        self.vib = CausalVIB(input_dim, latent_dim)
        self.context_gate = ContextInjectionGate(latent_dim) # Local-only
        self.audio_proj = nn.Linear(input_dim, latent_dim)
        
        # --- Stage 3: Interaction Reasoning ---
        self.cross_modal_block = CrossModalAttentionBlock(latent_dim) # Vis-Aud only
        
        # NEW: Transformer with Decoupled Modulation
        self.temporal_encoder = GammaGatedTransformer(
            dim=latent_dim, 
            style_dim=self.style_dim, 
            depth=2, drop_path=0.1
        )
        
        # --- Stage 4: Pooling & Classify ---
        self.temporal_pool = LearnedQueryAttention(input_dim=latent_dim, hidden_dim=128)
        self.contrastive_head = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, 128))
        
        fusion_dim = latent_dim + 1 
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 1024), 
            nn.LayerNorm(1024), nn.GELU(), nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
        self.cot_aligner = nn.Linear(latent_dim, input_dim)

    def forward_with_gamma(self, local_p, global_v, env_feat, audio_seq, gamma, mixup_alpha=None, targets=None):
        B, K, T, D = local_p.shape
        
        # 1. Spatial Pooling for Global
        if global_v.dim() == 3 and global_v.shape[1] > T:
            spatial_patches = global_v.shape[1] // T
            global_v_spatial = global_v.view(B * T, spatial_patches, D)
            pooled_spatial, _ = self.spatial_pool(global_v_spatial)
            global_v = pooled_spatial.view(B, T, D)
            
        # 2. Compute Global Style Token
        global_feat_avg = global_v.mean(dim=1) 
        style_token = self.global_style_proj(global_feat_avg) # [B, style_dim]
        scene_scale = self.scene_scaler(style_token)
        
        # 3. Disentangle
        vib_out = self.vib(local_p.view(-1, D))
        z_aff = vib_out['z_aff'].view(B, K, T, -1)
        z_env = vib_out['z_env'].view(B, K, T, -1)
        
        # 4. Mixup
        if self.training and mixup_alpha is not None and targets is not None:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            index = torch.randperm(B).to(local_p.device)
            z_aff = lam * z_aff + (1 - lam) * z_aff[index]
            z_env = lam * z_env + (1 - lam) * z_env[index]
            style_token = lam * style_token + (1 - lam) * style_token[index] 
            gamma = lam * gamma + (1 - lam) * gamma[index]
            scene_scale = lam * scene_scale + (1 - lam) * scene_scale[index]
            targets_a, targets_b = targets, targets[index]
            mixup_data = (targets_a, targets_b, lam)
        else:
            mixup_data = None

        # 5. Local Context Gating
        z_aff_context = self.context_gate(z_aff, z_env)
        
        # 6. Cross-Modal Attention
        vis_tokens = z_aff_context.permute(0, 2, 1, 3).reshape(B, T*K, -1)
        aud_tokens = self.audio_proj(audio_seq)
        refined_vis = self.cross_modal_block(vis_tokens, aud_tokens)
        
        # 7. Global-Adaptive Temporal Encoding
        refined_vis = refined_vis.view(B, T, K, -1).mean(dim=2) 
        gamma_prime = gamma * scene_scale 
        
        # Pass Global Style to DecoupledAdaLN
        h_seq = self.temporal_encoder(refined_vis, gamma_prime, style_token)
        
        # 8. Output
        h_final, attn_weights = self.temporal_pool(h_seq) 
        proj_feat = self.contrastive_head(h_final)
        rationale_proj = self.cot_aligner(h_final)
        
        # Handle mismatch in gamma for final fusion as well
        if gamma_prime.shape[1] != h_final.shape[1] and gamma_prime.shape[1] > 1:
             # Just pick the last value or mean if shapes don't align for fusion (usually not an issue for classification head input)
             # But for safety, we usually fuse a global scalar summary of gamma
             pass
        
        # For fusion, we just take the mean gamma over time to get a single scalar per video
        gamma_scalar = gamma_prime.mean(dim=1) 
        
        fusion_input = torch.cat([h_final, gamma_scalar], dim=-1)
        logits = self.classifier(fusion_input)
        
        return {
            'logits': logits, 'attn_weights': attn_weights, 'features': proj_feat, 
            'h_final': h_final, 'vib_out': vib_out, 'rationale': rationale_proj, 'mixup_data': mixup_data
        }