import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalSmoothingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Target: Pos(0), Neu(1), Neg(2)
        self.adj = torch.tensor([
            [0.85, 0.15, 0.00], 
            [0.10, 0.80, 0.10], 
            [0.00, 0.15, 0.85]  
        ])
            
    def forward(self, logits, target_indices):
        self.adj = self.adj.to(logits.device)
        soft_targets = self.adj[target_indices]
        log_probs = F.log_softmax(logits, dim=-1)
        return -(soft_targets * log_probs).sum(dim=-1).mean()


class VIBE_Loss(nn.Module):
    def __init__(self, lambda_sat=0.5, lambda_ortho=0.1):
        super().__init__()
        self.hierarchical_ce = HierarchicalSmoothingLoss()
        
        # Hyperparameters for structural regularization
        self.weights = {
            'sat': lambda_sat, 
            'ortho': lambda_ortho
        }

    def forward(self, output, targets, batch):
        mixup_data = output.get('mixup_data')
        
        # 1. Hierarchical Label Smoothing Loss - L_HCE
        if mixup_data is not None:
            targets_a, targets_b, lam = mixup_data
            loss_a = self.hierarchical_ce(output['logits'], targets_a)
            loss_b = self.hierarchical_ce(output['logits'], targets_b)
            l_hce = lam * loss_a + (1 - lam) * loss_b
        else:
            l_hce = self.hierarchical_ce(output['logits'], targets)
        
        # 2. VIB KL Loss - L_KL
        vib = output['vib_out']
        l_kl = (-0.5 * torch.sum(1 + vib['logvar_aff'] - vib['mu_aff'].pow(2) - vib['logvar_aff'].exp())) / targets.size(0)
        
        # 3. Semantic Alignment Loss - L_sat
        rationale_norm = F.normalize(output['rationale'], dim=-1)
        text_anch_norm = F.normalize(batch['text_anch'], dim=-1)
        l_sat = 1.0 - F.cosine_similarity(rationale_norm, text_anch_norm).mean()
        
        # 4. Orthogonality Loss - L_ortho
        z_a = vib['z_aff'].flatten(1)
        z_e = vib['z_env'].flatten(1)
        sim = F.cosine_similarity(z_a, z_e, dim=-1)
        l_ortho = (sim ** 2).mean()

        # 5. Total Objective - L_total
        total = l_hce + \
                (self.weights['sat'] * l_sat) + \
                (self.weights['ortho'] * l_ortho) + \
                l_kl

        return total, l_hce, l_sat, l_ortho, l_kl