import os
import torch
import torch.nn.functional as F
import numpy as np
import yaml
from pathlib import Path
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from data_preprocessing.preprocessing import VGAFDataPreprocessor, HubertExtractor, IdentityTracker, TubeExtractor
from data_preprocessing.extract_primitives_from_features import VIBEFeatureEngine
from model.vibe_model import VIBE_Transformer

class VIBEInferencer:
    """
    End-to-End Inference wrapper for VIBE model.
    Takes a raw video, runs data preprocessing, feature extraction, and model prediction.
    """
    def __init__(self, config_path, checkpoint_path=None, device='cuda'):
        # 1. Load config
        with open(config_path, 'r') as file:
            self.cfg = yaml.safe_load(file)
            
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Determine checkpoint path (CLI overrides config)
        ckpt_path = checkpoint_path or self.cfg.get('checkpoint_path')
        if not ckpt_path:
            raise ValueError("Checkpoint path must be provided either via CLI or in the YAML config under 'checkpoint_path'.")

        
        # 2. Init Preprocessors
        print("Initializing Trackers and Extractors...")
        self.tracker = IdentityTracker(smooth_window=5, device=self.device.type)
        self.tube_extractor = TubeExtractor(crop_size=(224, 224))
        self.hubert = HubertExtractor(device=self.device.type)
        
        # Helper to load frames (from preprocessor logic)
        self.preprocessor = VGAFDataPreprocessor(dataset_root="./", output_root="./", device=self.device.type)
        
        # 3. Init Primitive Extraction Engine (DINO, VMAE, RoBERTa)
        print("Initializing Feature Engines for Primitives...")
        self.feature_engine = VIBEFeatureEngine(device=self.device.type)
        
        # 4. Init VIBE Model
        print("Initializing VIBE Multi-Modal Framework...")
        self.model = VIBE_Transformer(
            input_dim=self.cfg.get('input_dim', 768), 
            latent_dim=self.cfg.get('latent_dim', 512),
            num_classes=self.cfg.get('num_classes', 3)
        ).to(self.device).eval()
        
        # Load weights
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint.get('state_dict', checkpoint))
        print(f"Model successfully loaded from {ckpt_path}")

    def process_video(self, video_path):
        video_path = Path(video_path)
        
        # 1. Load and Sample (6 fps constraint default)
        video_frames = self.preprocessor.load_and_sample_video(video_path)
        if video_frames is None:
            raise ValueError("Failed to load video frames.")
        
        H, W = video_frames[0].shape[:2]

        # 2. Audio Processing
        audio_features = self.hubert.extract_features(video_path)
        if audio_features is None:
            audio_features = np.zeros((1, 768)) # dummy fallback
            
        # 3. Track People
        trajectories = self.tracker.track_video(video_frames)
        if not trajectories:
            raise ValueError("No people detected in video!")

        # 4. Extract Visual Tubes
        video_tensor = torch.from_numpy(video_frames).permute(3, 0, 1, 2).float().to(self.device) / 255.0
        tubes_raw = self.tube_extractor.extract_tubes(video_tensor, trajectories, self.device)
        
        video_tensor = video_tensor.cpu()
        torch.cuda.empty_cache()

        # Format inputs for VIBEFeatureEngine
        full_frames_np = video_frames
        
        persons = []
        raw_coords_dict = {}
        for pid, tube in tubes_raw.items():
            visual_uint8 = (tube.permute(1, 2, 3, 0).cpu() * 255).byte().numpy()
            persons.append(visual_uint8)
            raw_coords = np.array([t[1] for t in trajectories[pid]], dtype=np.float32)
            raw_coords_dict[int(pid)] = raw_coords

        # 5. Extract Primitives (No text required for inference)
        primitives = self.feature_engine.extract_primitives(
            full_frames_np, 
            persons, 
            raw_coords_dict, 
            audio_features, 
            "dummy inference text" # The extraction script needs a string but the model forward pass won't use it
        )
        return primitives

    def format_tensors(self, primitives):
        """ Format primitives into model inputs for Inference """
        max_k = self.cfg.get('max_k', 8)
        max_t = self.cfg.get('max_t', 32)
        
        # Global V
        global_v = torch.from_numpy(primitives['global_v_seq']).squeeze(0).float()
        # Env Feat
        env_feat = torch.from_numpy(primitives['env_feat']).squeeze(0).float()
        # Text Anch
        text_anch = torch.from_numpy(primitives['text_anch']).squeeze(0).float()
        
        # Audio
        audio_seq = torch.from_numpy(primitives['audio_hubert']).float()
        if audio_seq.shape[0] > max_t:
            audio_seq = audio_seq[:max_t]
        elif audio_seq.shape[0] < max_t:
            pad_a = torch.zeros(max_t - audio_seq.shape[0], 768)
            audio_seq = torch.cat([audio_seq, pad_a], dim=0)

        # Local P
        person_seqs = primitives['local_p_seqs']
        formatted_p = []
        for p_data_np in person_seqs[:max_k]:
            p_data = torch.from_numpy(p_data_np).squeeze(0).float() # [T, 768]
            curr_t = p_data.shape[0]
            if curr_t > max_t:
                p_data = p_data[:max_t]
            elif curr_t < max_t:
                padding = torch.zeros(max_t - curr_t, 768)
                p_data = torch.cat([p_data, padding], dim=0)
            formatted_p.append(p_data)
            
        while len(formatted_p) < max_k:
            formatted_p.append(torch.zeros(max_t, 768))
        local_p_tensor = torch.stack(formatted_p)

        # Gamma (Synchrony)
        sync_mat = primitives['physics_sync']
        padded_sync = np.zeros((max_k, max_k))
        k_limit = min(sync_mat.shape[0], max_k)
        padded_sync[:k_limit, :k_limit] = sync_mat[:k_limit, :k_limit]
        gamma_tensor = torch.from_numpy(padded_sync).float()

        return {
            'global_v': global_v.unsqueeze(0).to(self.device),
            'env_feat': env_feat.unsqueeze(0).to(self.device),
            'audio_seq': audio_seq.unsqueeze(0).to(self.device),
            'local_p': local_p_tensor.unsqueeze(0).to(self.device),
            'gamma': gamma_tensor.unsqueeze(0).to(self.device)
        }

    @torch.no_grad()
    def infer(self, video_path):
        print(f"--- Running Inference on {video_path} ---")
        primitives = self.process_video(video_path)
        inputs = self.format_tensors(primitives)

        
        # Forward pass
        out = self.model.forward_with_gamma(
            inputs['local_p'], 
            inputs['global_v'],
            inputs['env_feat'], 
            inputs['audio_seq'],
            inputs['gamma']
        )
        
        logits = out['logits']
        probs = F.softmax(logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=-1).item()
        
        emotion_map = {0: 'Positive', 1: 'Neutral', 2: 'Negative'}
        emotion = emotion_map.get(pred_idx, "Unknown")
        
        return {
            "prediction": emotion,
            "probabilities": probs.cpu().numpy().tolist()[0],
            "reasoning_rationale": out['rationale'].cpu().numpy()
        }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="End-To-End VIBE Inference")
    parser.add_argument("--video", type=str, required=True, help="Path to input video (.mp4/.avi)")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--checkpoint", type=str, required=False, help="Model Checkpoint path (Overrides YAML config)")
    
    args = parser.parse_args()
    
    inferencer = VIBEInferencer(config_path=args.config, checkpoint_path=args.checkpoint)
    result = inferencer.infer(args.video)
    
    print("\n" + "="*50)
    print(" INFERENCE RESULT ")
    print("="*50)
    print(f"  > Predicted Emotion : {result['prediction']}")
    print(f"  > Probabilities     : Positive({result['probabilities'][0]:.3f}) | Neutral({result['probabilities'][1]:.3f}) | Negative({result['probabilities'][2]:.3f})")
    print("="*50)
