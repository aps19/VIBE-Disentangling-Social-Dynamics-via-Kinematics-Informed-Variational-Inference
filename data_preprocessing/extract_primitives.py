import os
import h5py
import torch
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import (
    VideoMAEImageProcessor, VideoMAEModel,
    AutoImageProcessor, Dinov2Model,
    RobertaTokenizer, RobertaModel
)

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VIBEFeatureEngine:
    def __init__(self, device='cuda'):
        self.device = device
        
        # 1. Global Context: VideoMAE V2 (Sequential Tokens)
        logger.info("Loading VideoMAE V2...")
        self.vmae_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
        self.vmae_model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(device).eval()
        
        # 2. Local/Env Affect: DINOv2 (Spatial-Temporal Tokens)
        logger.info("Loading DINOv2...")
        self.dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        self.dino_model = Dinov2Model.from_pretrained("facebook/dinov2-base").to(device).eval()
        
        # 3. Language Anchor: RoBERTa
        logger.info("Loading RoBERTa...")
        self.text_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.text_model = RobertaModel.from_pretrained("roberta-base").to(device).eval()

    def calculate_physics_sync(self, trajectories, frame_dim, max_k=5):
        """Calculates Behavioral Synchrony Matrix from raw_coords."""
        H, W = frame_dim
        pids = list(trajectories.keys())[:max_k]
        vel_list = []
        for pid in pids:
            coords = trajectories[pid]
            # Calculate Centroids
            cx = (coords[:, 0] + coords[:, 2]/2) / W
            cy = (coords[:, 1] + coords[:, 3]/2) / H
            
            # Calculate Velocity (diff)
            if len(cx) > 1:
                v = np.stack([np.diff(cx), np.diff(cy)], axis=1)
                # Normalize
                v_norm = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-6)
                vel_list.append(v_norm)
            else:
                # Handle single-frame tracks
                vel_list.append(np.zeros((0, 2)))

        sync_mat = np.zeros((max_k, max_k))
        for i in range(len(vel_list)):
            for j in range(i + 1, len(vel_list)):
                min_len = min(len(vel_list[i]), len(vel_list[j]))
                if min_len > 0:
                    corr = np.mean(np.sum(vel_list[i][:min_len] * vel_list[j][:min_len], axis=1))
                    sync_mat[i, j] = sync_mat[j, i] = corr
        return sync_mat

    @torch.no_grad()
    def extract_primitives(self, full_frames, persons, trajectories, audio_feat, description):
        """
        Extracts temporal sequences for C-SYNC.
        """
        # Use Mixed Precision for Speed
        with torch.cuda.amp.autocast():
            # --- 1. Global Scene Sequence (VideoMAE) ---
            if len(full_frames) > 0:
                idx = np.linspace(0, len(full_frames)-1, 16).astype(int)
                v_in = self.vmae_processor([full_frames[i] for i in idx], return_tensors="pt").to(self.device)
                global_v_seq = self.vmae_model(**v_in).last_hidden_state.cpu().numpy()
            else:
                global_v_seq = np.zeros((1, 1568, 768))

            # --- 2. Environmental Prior (DINOv2 Global) ---
            if len(full_frames) > 0:
                mid_frame = full_frames[len(full_frames)//2]
                env_in = self.dino_processor(mid_frame, return_tensors="pt").to(self.device)
                env_feat = self.dino_model(**env_in).last_hidden_state[:, 0, :].cpu().numpy()
            else:
                env_feat = np.zeros((1, 768))

            # --- 3. Local Affective Sequences (DINOv2 Local) - BATCHED ---
            local_p_seqs = []
            
            # Collect all frames from all persons to batch process
            all_person_frames = []
            person_lengths = []
            
            valid_persons = persons[:min(len(persons), 5)]
            
            for p_frames in valid_persons:
                if len(p_frames) > 0:
                    all_person_frames.extend(list(p_frames))
                    person_lengths.append(len(p_frames))
                else:
                    person_lengths.append(0)
            
            if all_person_frames:
                # Process in batches to avoid OOM but maximize throughput
                batch_size = 32 
                all_feats = []
                
                for i in range(0, len(all_person_frames), batch_size):
                    batch = all_person_frames[i:i+batch_size]
                    p_in = self.dino_processor(batch, return_tensors="pt").to(self.device)
                    p_out = self.dino_model(**p_in).last_hidden_state[:, 0, :]
                    all_feats.append(p_out.cpu())
                
                all_feats = torch.cat(all_feats, dim=0).numpy()
                
                # Split back into per-person sequences
                cursor = 0
                for length in person_lengths:
                    if length > 0:
                        local_p_seqs.append(all_feats[cursor:cursor+length])
                        cursor += length
                    else:
                        local_p_seqs.append(np.zeros((1, 768)))
            else:
                # No valid person frames
                for _ in range(len(valid_persons)):
                    local_p_seqs.append(np.zeros((1, 768)))

            # --- 4. Physics and Text ---
            physics_sync = self.calculate_physics_sync(trajectories, full_frames.shape[1:3] if len(full_frames) > 0 else (224, 224))
            
            t_in = self.text_tokenizer(str(description), return_tensors="pt", padding=True, truncation=True).to(self.device)
            text_anch = self.text_model(**t_in).pooler_output.cpu().numpy()

        return {
            'global_v_seq': global_v_seq,
            'env_feat': env_feat,
            'local_p_seqs': local_p_seqs,
            'physics_sync': physics_sync,
            'text_anch': text_anch,
            'audio_hubert': audio_feat
        }

def run_vibe_extraction(source_root, output_root, train_xlsx, val_xlsx):
    engine = VIBEFeatureEngine()
    
    for split, xlsx_path in [('train', train_xlsx), ('val', val_xlsx)]:
        logger.info(f"Processing Split: {split.upper()}")
        
        if not os.path.exists(xlsx_path):
            logger.warning(f"Excel file not found: {xlsx_path}. Skipping {split}.")
            continue
            
        df = pd.read_excel(xlsx_path)
        dest_dir = Path(output_root) / split
        dest_dir.mkdir(parents=True, exist_ok=True)
        source_dir = Path(source_root) / 'data' / split
        
        if not source_dir.exists():
            logger.warning(f"Source directory not found: {source_dir}. Have you run the preprocessor?")
            continue

        processed_count = 0
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"VIBE Extraction ({split})"):
            vid_id = str(row['Vid_name']).strip()
            h5_in_file = source_dir / f"{vid_id}.h5"
            
            if not h5_in_file.exists(): 
                continue
            
            try:
                with h5py.File(h5_in_file, 'r') as f_in:
                    frames = f_in['full_frames'][:]
                    audio = f_in['audio_features'][:]
                    
                    # Load Persons
                    pids = sorted(f_in['persons'].keys(), key=lambda x: int(x))
                    persons = [f_in[f'persons/{pid}/visual'][:] for pid in pids]
                    trajectories = {int(pid): f_in[f'persons/{pid}/raw_coords'][:] for pid in pids}
                    
                    primitives = engine.extract_primitives(frames, persons, trajectories, audio, str(row['Description']))
                    
                    # Save as Sequential HDF5
                    with h5py.File(dest_dir / f"{vid_id}_vibe.h5", 'w') as f_out:
                        f_out.create_dataset('global_v_seq', data=primitives['global_v_seq'], compression='gzip')
                        f_out.create_dataset('env_feat', data=primitives['env_feat'], compression='gzip')
                        f_out.create_dataset('physics_sync', data=primitives['physics_sync'], compression='gzip')
                        f_out.create_dataset('audio_seq', data=primitives['audio_hubert'], compression='gzip')
                        f_out.create_dataset('text_anch', data=primitives['text_anch'], compression='gzip')
                        
                        p_group = f_out.create_group('person_sequences')
                        for i, p_seq in enumerate(primitives['local_p_seqs']):
                            p_group.create_dataset(f'p_{i}', data=p_seq, compression='gzip')
                        
                        f_out.attrs['label'] = row['Label']
                        processed_count += 1
            except Exception as e:
                logger.error(f"Failed to process {vid_id}: {e}")
                continue
        
        logger.info(f"Finished {split}. Processed {processed_count} videos.")

if __name__ == "__main__":
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description="Extract VIBE primitives")
    parser.add_argument("--config", type=str, required=True, help="Path to config file (e.g., config/data_extraction.yaml)")
    parser.add_argument("--dataset", type=str, choices=["vgaf", "gecv"], required=True, help="Dataset to process (vgaf or gecv)")
    args = parser.parse_args()
    
    # Load YAML Configuration
    with open(args.config, 'r') as file:
        cfg = yaml.safe_load(file)
        
    dataset_cfg = cfg.get(args.dataset)
    if not dataset_cfg:
        raise ValueError(f"Configuration for dataset '{args.dataset}' not found in {args.config}")
        
    logger.info(f"Starting extraction for {args.dataset.upper()} dataset")
    run_vibe_extraction(
        dataset_cfg['raw_data_root'], 
        dataset_cfg['vibe_output_root'], 
        dataset_cfg['train_xlsx'], 
        dataset_cfg['val_xlsx']
    )