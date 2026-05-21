from sklearn.metrics import accuracy_score, classification_report, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path
import numpy as np

from model.vibe_model import VIBE_Transformer
from dataloading.vibe_dataset import VIBEDataset, csync_collate_fn, create_group_split
from vizualization import VIBE_DeepVisualizer
from losses.loss_function import VIBE_Loss

class VIBETrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup Directories
        self.log_dir = Path(cfg['log_dir'])
        self.ckpt_dir = Path(cfg['checkpoint_dir'])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.viz = VIBE_DeepVisualizer(str(self.log_dir))
        
        print("Initializing Datasets...")
        
        # --- A. Perform Group-Aware Split on 'train' folder ---
        train_files, val_files = create_group_split(
            cfg['data_root'], 
            val_ratio=cfg.get('val_ratio', 0.1)
        )
        
        # Extract dataset params
        max_k = cfg.get('max_k', 5)
        max_t = cfg.get('max_t', 32)
        num_workers = cfg.get('num_workers', 0)
        
        self.train_loader = DataLoader(
            VIBEDataset(train_files, max_k=max_k, max_t=max_t), 
            batch_size=cfg['batch_size'], shuffle=True, 
            collate_fn=csync_collate_fn, num_workers=num_workers, pin_memory=True
        )
        self.val_loader = DataLoader(
            VIBEDataset(val_files, max_k=max_k, max_t=max_t), 
            batch_size=cfg['batch_size'], shuffle=False, 
            collate_fn=csync_collate_fn, num_workers=num_workers, pin_memory=True
        )
        
        # --- B. Load Original Validation Set as 'Test' ---
        test_source = Path(cfg['data_root']) / 'val'
        test_files = sorted(list(test_source.glob("*_csync.h5")))
        self.test_loader = DataLoader(
            VIBEDataset(test_files, max_k=max_k, max_t=max_t),
            batch_size=cfg['batch_size'], shuffle=False,
            collate_fn=csync_collate_fn, num_workers=num_workers
        )
        print(f"  > Test Set (Held Out): {len(test_files)} clips loaded.")

        # --- C. Init Model (V5.3 Global-Aware) ---
        print(f"Initializing VIBE (Latent: {cfg['latent_dim']})...")
        self.model = VIBE_Transformer(
            input_dim=cfg.get('input_dim', 768), 
            latent_dim=cfg['latent_dim'],
            num_classes=cfg.get('num_classes', 3)
        ).to(self.device)
        
        # --- D. Init Loss (V5.5 Improved) ---
        self.criterion = VIBE_Loss(
            lambda_supcon=cfg.get('lambda_supcon', 0.5),
            lambda_cot=cfg.get('lambda_cot', 0.5),
            lambda_ortho=cfg.get('lambda_ortho', 0.1) # New Ortho Weight
        ).to(self.device)
        
        # --- E. Optimizer & Scheduler ---
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=cfg['lr'],
            weight_decay=cfg.get('weight_decay', 0.01)
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max', 
            patience=cfg.get('scheduler_patience', 3),
            factor=cfg.get('scheduler_factor', 0.1)
        )

    def run(self, epochs):
        print(f"Starting VIBE Training on {self.device}")
        print(f"Checkpoints will be saved to: {self.ckpt_dir}")
        best_acc = 0.0
        
        mixup_alpha = self.cfg.get('mixup_alpha', 0.2)
        grad_clip = self.cfg.get('grad_clip', 1.0)
        save_threshold = self.cfg.get('save_threshold', 0.60)
        
        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0
            pbar = tqdm(self.train_loader, desc=f"Ep {epoch}")
            
            for batch in pbar:
                if batch is None: continue
                
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)
                
                # Model Forward
                out = self.model.forward_with_gamma(
                    local_p=batch['local_p'], 
                    global_v=batch['global_v'],
                    env_feat=batch['env_feat'], 
                    audio_seq=batch['audio_seq'],
                    gamma=batch['gamma'],
                    mixup_alpha=mixup_alpha,
                    targets=batch['labels']
                )
                
                # Loss Calculation (Unpacking Ortho Loss now)
                loss, l_ce, l_sup, l_ortho = self.criterion(
                    out, batch['labels'], batch
                )
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.optimizer.step()
                
                total_loss += loss.item()
                
                # Monitor Disentanglement (Ortho) in realtime
                pbar.set_postfix({
                    'L': f"{loss.item():.3f}", 
                    'CE': f"{l_ce.item():.3f}",
                    'Ort': f"{l_ortho.item():.3f}"
                })
            
            metrics = self.validate(epoch, self.val_loader, mode='Val')
            self.scheduler.step(metrics['acc'])
            
            # Save Checkpoints
            checkpoint = {
                'epoch': epoch, 'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(), 'best_acc': best_acc, 'config': self.cfg
            }
            torch.save(checkpoint, self.ckpt_dir / "checkpoint_last.pth")
            
            if metrics['acc'] > save_threshold:
                ckpt_name = f"model_ep{epoch}_acc{metrics['acc']:.4f}_f1{metrics['f1']:.4f}.pth"
                torch.save(self.model.state_dict(), self.ckpt_dir / ckpt_name)
                print(f"   >>> Model Saved (Acc > {save_threshold:.0%}): {ckpt_name}")
            
            if metrics['acc'] > best_acc:
                best_acc = metrics['acc']
                torch.save(self.model.state_dict(), self.ckpt_dir / "model_best.pth")
                print(f"   >>> New Best Model Saved (Acc: {best_acc:.4f})")

        # Final Test
        print("\n\n" + "="*50)
        print("  FINAL EVALUATION ON HELD-OUT TEST SET")
        print("="*50)
        best_ckpt = torch.load(self.ckpt_dir / "model_best.pth")
        self.model.load_state_dict(best_ckpt)
        self.validate(epochs, self.test_loader, mode='Test')
        self.writer.close()

    @torch.no_grad()
    def validate(self, epoch, loader, mode='Val'):
        self.model.eval()
        preds, targets = [], []
        
        # Storage for Visualization
        rationale_list, text_list = [], []
        h_final_list = []
        z_aff_list, z_env_list = [], []
        gamma_list = []
        
        for batch in tqdm(loader, desc=f"Validating {mode}", leave=False):
            if batch is None: continue
            
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device)
            
            out = self.model.forward_with_gamma(
                batch['local_p'], batch['global_v'],
                batch['env_feat'], batch['audio_seq'],
                batch['gamma']
            )
            
            preds.extend(torch.argmax(out['logits'], dim=1).cpu().numpy())
            targets.extend(batch['labels'].cpu().numpy())
            
            # Subsample for Viz
            if len(h_final_list) * self.cfg['batch_size'] < 800:
                rationale_list.append(out['rationale'].cpu())
                text_list.append(batch['text_anch'].cpu())
                h_final_list.append(out['h_final'].cpu())
                
                # Reshape VIB outputs
                B, K, T, _ = batch['local_p'].shape
                z_aff_flat = out['vib_out']['z_aff']
                z_env_flat = out['vib_out']['z_env']
                
                z_aff_reshaped = z_aff_flat.view(B, K, T, -1)
                z_env_reshaped = z_env_flat.view(B, K, T, -1)
                
                z_aff_list.append(z_aff_reshaped.mean(dim=(1, 2)).cpu())
                z_env_list.append(z_env_reshaped.mean(dim=(1, 2)).cpu())
                gamma_list.append(batch['gamma'].cpu())

        if len(targets) == 0: return {'acc': 0, 'f1': 0}
        
        acc = accuracy_score(targets, preds)
        f1 = f1_score(targets, preds, average='weighted', zero_division=0)
        
        # --- VISUALIZATION BLOCK ---
        if mode == 'Val' and len(rationale_list) > 0:
            gammas = torch.cat(gamma_list).numpy()
            targets_viz = np.array(targets)[:len(gammas)]
            z_aff = torch.cat(z_aff_list).numpy()
            z_env = torch.cat(z_env_list).numpy()
            texts = torch.cat(text_list).numpy()
            rationales = torch.cat(rationale_list).numpy()
            
            # Metrics
            ebs = F.cosine_similarity(torch.tensor(z_aff), torch.tensor(z_env)).abs().mean().item()
            rc = F.cosine_similarity(torch.tensor(rationales), torch.tensor(texts)).mean().item()
            gamma_mean = gammas.mean().item()
            
            self.viz.update_history(epoch, ebs, rc, acc, gamma_mean)
            
            if epoch % 2 == 0 or epoch == 1:
                self.viz.plot_disentanglement_proof(z_aff, z_env, targets_viz, epoch)
                self.viz.plot_gamma_dynamics(gammas, targets_viz, epoch)
                self.viz.plot_3d_manifold(torch.cat(h_final_list).numpy(), targets_viz, epoch)
                self.viz.plot_cot_alignment(rationales, texts, epoch)
                self.viz.plot_confusion_matrix(np.array(targets), np.array(preds), epoch)
                self.viz.plot_metric_trends()

            print("\n" + "="*50)
            print(f"  {mode} Report | Epoch {epoch}")
            print("="*50)
            print(classification_report(targets, preds, target_names=['Pos', 'Neu', 'Neg'], zero_division=0))
            print("-" * 50)
            print(f"  > Accuracy             : {acc:.4f}")
            print(f"  > F1 Score (Weighted)  : {f1:.4f}")
            print(f"  > EBS (Bias Score)     : {ebs:.4f}")
            print("="*50 + "\n")
            
            self.writer.add_scalar(f'{mode}/Accuracy', acc, epoch)
            self.writer.add_scalar(f'{mode}/F1_Score', f1, epoch)
            self.writer.add_scalar(f'{mode}/EBS', ebs, epoch)
        
        elif mode == 'Test':
             print("\n" + "="*50)
             print(f"  FINAL TEST REPORT")
             print("="*50)
             print(classification_report(targets, preds, target_names=['Pos', 'Neg', 'Neu'], zero_division=0))
             print("-" * 50)
             print(f"  > Accuracy             : {acc:.4f}")
             print(f"  > F1 Score (Weighted)  : {f1:.4f}")
             print("="*50)

        return {'acc': acc, 'f1': f1}