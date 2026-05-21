import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
import logging
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


class VIBEDataset(Dataset):
    def __init__(self, file_list, max_k=5, max_t=32):
        """
        Modified to accept a direct list of file paths.
        This allows complex splitting logic (like Group K-Fold) to happen outside.
        """
        self.files = file_list
        self.max_k = max_k 
        self.max_t = max_t 
        self.label_map = {1: 0, 2: 1, 3: 2}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        h5_path = self.files[idx]
        try:
            with h5py.File(h5_path, 'r') as f:
                # Load features 
                global_v = torch.from_numpy(f['global_v_seq'][:]).squeeze(0)
                env_feat = torch.from_numpy(f['env_feat'][:]).squeeze(0)
                text_anch = torch.from_numpy(f['text_anch'][:]).squeeze(0)

                person_seqs = []V_Los
                p_group = f['person_sequences']
                p_keys = sorted(p_group.keys(), key=lambda x: int(x.split('_')[1]))
                
                for pid in p_keys[:self.max_k]:
                    p_data = torch.from_numpy(p_group[pid][:]) 
                    curr_t = p_data.shape[0]
                    if curr_t > self.max_t:
                        p_data = p_data[:self.max_t]
                    elif curr_t < self.max_t:
                        padding = torch.zeros(self.max_t - curr_t, 768)
                        p_data = torch.cat([p_data, padding], dim=0)
                    person_seqs.append(p_data)

                real_k = len(person_seqs)
                while len(person_seqs) < self.max_k:
                    person_seqs.append(torch.zeros(self.max_t, 768))
                local_p_tensor = torch.stack(person_seqs)

                sync_mat = torch.from_numpy(f['physics_sync'][:])
                padded_sync = torch.zeros(self.max_k, self.max_k)
                k_limit = min(sync_mat.shape[0], self.max_k)
                padded_sync[:k_limit, :k_limit] = sync_mat[:k_limit, :k_limit]

                audio_seq = torch.from_numpy(f['audio_seq'][:])
                if audio_seq.shape[0] > self.max_t:
                    audio_seq = audio_seq[:self.max_t]
                elif audio_seq.shape[0] < self.max_t:
                    pad_a = torch.zeros(self.max_t - audio_seq.shape[0], 768)
                    audio_seq = torch.cat([audio_seq, pad_a], dim=0)

                label = self.label_map.get(int(f.attrs['label']), 2)

            return {
                'local_p': local_p_tensor.float(), 'global_v': global_v.float(), 
                'env_feat': env_feat.float(), 'audio_seq': audio_seq.float(),
                'sync_mat': padded_sync.float(), 'text_anch': text_anch.float(),
                'label': torch.tensor(label, dtype=torch.long),
                'real_k': torch.tensor(real_k, dtype=torch.long)
            }
        except Exception as e:
            print(f"Error loading {h5_path}: {e}")
            return None

def compute_masked_gamma(sync_mat, real_k):
    """
    FIX APPLIED: Computes Gamma only over active agents to prevent dilution.
    """
    B = sync_mat.shape[0]
    gammas = []
    for i in range(B):
        k = real_k[i].item()
        if k > 1: # Need at least 2 people for synchrony
            valid_sync = sync_mat[i, :k, :k]
            # Exclude diagonal (self-sync) for purer group metric
            mask = ~torch.eye(k, dtype=torch.bool)
            gammas.append(valid_sync[mask].mean())
        else:
            gammas.append(torch.tensor(0.0))
    return torch.stack(gammas).view(B, 1, 1)

def csync_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0: return None

    global_v = torch.stack([b['global_v'] for b in batch])
    env_feat = torch.stack([b['env_feat'] for b in batch])
    local_p = torch.stack([b['local_p'] for b in batch])
    sync_mat = torch.stack([b['sync_mat'] for b in batch])
    text_anch = torch.stack([b['text_anch'] for b in batch])
    audio_seq = torch.stack([b['audio_seq'] for b in batch])
    labels = torch.stack([b['label'] for b in batch])
    real_k = torch.stack([b['real_k'] for b in batch])

    gamma = compute_masked_gamma(sync_mat, real_k)

    return {
        'global_v': global_v,
        'env_feat': env_feat,
        'local_p': local_p,
        'sync_mat': sync_mat,
        'gamma': gamma,
        'text_anch': text_anch,
        'audio_seq': audio_seq,
        'labels': labels,
        'real_k': real_k
    }

import random
from collections import defaultdict
from pathlib import Path

def create_group_split(data_root_path, val_ratio=0.2):
    """
    Splits data such that all clips from the same video (Group) stay together.
    Prevents data leakage from training to validation.
    """
    train_source = Path(data_root_path) / 'train'
    all_files = sorted(list(train_source.glob("*_csync.h5")))
    
    # 1. Group files by Video ID (Prefix before the first underscore)
    # Example: '1_1_csync.h5' -> '1', '5_2_csync.h5' -> '5'
    video_groups = defaultdict(list)
    for f in all_files:
        # Extract ID: "5_1_csync.h5" -> "5"
        video_id = f.name.split('_')[0]
        video_groups[video_id].append(f)
    
    # 2. Shuffle Unique Video IDs
    unique_videos = list(video_groups.keys())
    random.shuffle(unique_videos)
    
    # 3. Split based on Video Count (not clip count)
    split_idx = int(len(unique_videos) * (1 - val_ratio))
    train_vids = unique_videos[:split_idx]
    val_vids = unique_videos[split_idx:]
    
    # 4. Flatten back to file lists
    train_files = [f for vid in train_vids for f in video_groups[vid]]
    val_files = [f for vid in val_vids for f in video_groups[vid]]
    
    print(f"Splitting Complete:")
    print(f"  > Unique Videos: {len(unique_videos)}")
    print(f"  > Training:   {len(train_vids)} videos -> {len(train_files)} clips")
    print(f"  > Validation: {len(val_vids)} videos -> {len(val_files)} clips")
    
    return train_files, val_files