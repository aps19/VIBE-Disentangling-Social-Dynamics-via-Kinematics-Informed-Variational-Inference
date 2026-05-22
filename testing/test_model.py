import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd 
from sklearn.metrics import accuracy_score, classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import plotly.express as px

from model.vibe_model import VIBE_Transformer
from dataloading.vibe_dataset import VIBEDataset, vibe_collate_fn

class CSYNCTester:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. Setup Directories
        self.log_dir = Path(cfg['log_dir'])
        self.vis_dir = self.log_dir / "comparative_test_results"
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. Initialize Visualizer (Assuming external class exists, otherwise we skip static viz)
        # self.viz = CSYNC_DeepVisualizer(str(self.log_dir)) 
        # self.viz.save_dir = self.vis_dir 

        print(f"Initializing Batch Test Environment on {self.device}...")

        # 3. Load Test Data
        test_source = Path(cfg['data_root']) / 'val'
        self.test_files = sorted(list(test_source.glob("*_csync.h5")))
        
        if len(self.test_files) == 0:
            raise ValueError(f"No test files found in {test_source}")
            
        self.test_loader = DataLoader(
            VIBEDataset(self.test_files), 
            batch_size=cfg['batch_size'], 
            shuffle=False, 
            collate_fn=vibe_collate_fn,
            num_workers=0
        )
        print(f"  > Loaded Held-Out Test Set: {len(self.test_files)} clips.")

        # 4. Initialize Architecture
        self.model = VIBE_Transformer(latent_dim=cfg['latent_dim']).to(self.device)

    def load_weights(self, checkpoint_path):
        """Helper to load weights without crashing."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
            return checkpoint.get('epoch', 'Unknown'), checkpoint.get('best_acc', 0.0)
        else:
            self.model.load_state_dict(checkpoint)
            return 'N/A', 0.0

    def evaluate_single_model(self, model_name):
        self.model.eval()
        preds, targets = [], []
        
        # Visualization buffers
        h_final_list = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc=f"Eval {model_name}", leave=False):
                if batch is None: continue
                
                # Move Inputs
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)
                
                # Forward Pass
                out = self.model.forward_with_gamma(
                    batch['local_p'], batch['global_v'], 
                    batch['env_feat'], batch['audio_seq'], 
                    batch['gamma']
                )
                
                preds.extend(torch.argmax(out['logits'], dim=1).cpu().numpy())
                targets.extend(batch['labels'].cpu().numpy())
                
                # Collect Embeddings (Limit to ~1000 samples to keep t-SNE fast)
                if len(h_final_list) * self.cfg['batch_size'] < 1000:
                    h_final_list.append(out['h_final'].cpu())

        acc = accuracy_score(targets, preds)
        f1 = f1_score(targets, preds, average='weighted', zero_division=0)
        
        return {
            'acc': acc, 
            'f1': f1, 
            'preds': preds, 
            'targets': targets,
            'h_final': h_final_list
        }

    def generate_interactive_3d_tsne(self, embeddings, labels, model_name):
        """
        Generates an interactive HTML 3D plot using Plotly.
        """
        print(f"Generating Interactive 3D Plot for {model_name}...")
        
        # 1. Prepare Data
        if isinstance(embeddings, list):
            embeddings = torch.cat(embeddings).numpy()
        
        # Truncate labels to match embeddings length (due to subsampling in eval loop)
        labels = np.array(labels[:len(embeddings)])
        
        # Map numeric labels to names
        label_map = {0: 'Positive', 1: 'Neutral', 2: 'Negative'}
        str_labels = [label_map.get(l, str(l)) for l in labels]

        # 2. Run t-SNE (3 Components)
        print("  > Running t-SNE (3 components)...")
        tsne = TSNE(n_components=3, perplexity=30, random_state=42)
        projections = tsne.fit_transform(embeddings)
        
        # 3. Create DataFrame for Plotly
        df = pd.DataFrame({
            'x': projections[:, 0],
            'y': projections[:, 1],
            'z': projections[:, 2],
            'Label': str_labels
        })
        
        # 4. Generate Plot
        fig = px.scatter_3d(
            df, x='x', y='y', z='z',
            color='Label',
            title=f"3D t-SNE Manifold: {model_name}",
            labels={'color': 'Emotion'},
            opacity=0.7,
            color_discrete_map={'Positive': 'green', 'Negative': 'red', 'Neutral': 'blue'}
        )
        
        fig.update_traces(marker=dict(size=5))
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=30))
        
        # 5. Save HTML
        save_path = self.vis_dir / f"Interactive_3D_{model_name}.html"
        fig.write_html(str(save_path))
        print(f"  > Interactive Plot Saved: {save_path}")

    def scan_and_evaluate_all(self, checkpoint_dir):
        ckpt_path = Path(checkpoint_dir)
        models = sorted(list(ckpt_path.glob("*.pth")))
        
        if not models:
            print(f"No models found in {ckpt_path}")
            return

        print("\n" + "="*60)
        print(f"  STARTING MODEL TOURNAMENT: {len(models)} Candidates")
        print("="*60)

        results = []
        best_acc = -1
        best_model_name = ""
        best_h_final = None
        best_targets = None

        for model_path in models:
            # 1. Load & Eval
            ep, saved_acc = self.load_weights(model_path)
            metrics = self.evaluate_single_model(model_path.name)
            
            print(f"[{model_path.name}] -> Acc: {metrics['acc']:.4f} | F1: {metrics['f1']:.4f}")
            
            results.append({
                'Model Name': model_path.name,
                'Epoch': ep,
                'Test Accuracy': metrics['acc'],
                'Test F1': metrics['f1'],
                'Saved Best Acc': saved_acc
            })
            
            # Track Winner
            if metrics['acc'] > best_acc:
                best_acc = metrics['acc']
                best_model_name = model_path.name
                best_h_final = metrics['h_final']
                best_targets = metrics['targets']

        # --- FINAL REPORT ---
        df = pd.DataFrame(results)
        df = df.sort_values(by='Test Accuracy', ascending=False).reset_index(drop=True)
        
        print("\n" + "="*60)
        print("  🏆 TOURNAMENT LEADERBOARD 🏆")
        print("="*60)
        print(df.to_string())
        
        df.to_csv(self.vis_dir / "model_comparison_results.csv", index=False)

        if best_h_final is not None:
             print(f"\nGeneratig Visuals for Winner: {best_model_name}")
             self.generate_interactive_3d_tsne(best_h_final, best_targets, best_model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test VIBE Framework")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    args = parser.parse_args()

    # Load YAML Configuration
    with open(args.config, 'r') as file:
        test_config = yaml.safe_load(file)
    
    tester = CSYNCTester(test_config)
    
    # Assuming 'checkpoint_folder' is defined in the config yaml
    tester.scan_and_evaluate_all(test_config['checkpoint_folder'])


