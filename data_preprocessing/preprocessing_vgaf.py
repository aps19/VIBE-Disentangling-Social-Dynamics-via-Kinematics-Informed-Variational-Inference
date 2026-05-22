import os
import json
import shutil
import logging
import math
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re

import numpy as np
import cv2
import h5py
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
import librosa

# --- FIX: Import FeatureExtractor instead of Processor ---
from transformers import HubertModel, Wav2Vec2FeatureExtractor

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Check for Ultralytics
try:
    from ultralytics import YOLO
except ImportError:
    logger.error("Ultralytics not found. Please run: pip install ultralytics")
    raise

# --- NEW: HuBERT Extractor (FIXED) ---
class HubertExtractor:
    """
    Extracts high-level audio features using a pre-trained HuBERT model.
    """
    def __init__(self, model_name: str = "facebook/hubert-base-ls960", device: str = 'cuda'):
        self.device = device
        self.target_sr = 16000 # HuBERT requires 16kHz
        
        logger.info(f"Loading HuBERT model: {model_name}...")
        try:
            # Use FeatureExtractor. The base model has no tokenizer (vocab).
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
            self.model = HubertModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to load HuBERT: {e}")
            raise

    @torch.no_grad()
    def extract_features(self, video_path: Path) -> Optional[np.ndarray]:
        try:
            # 1. Load Audio
            # librosa loads as float32 in range [-1, 1]
            y, sr = librosa.load(str(video_path), sr=self.target_sr, mono=True)
            
            if len(y) < 1600: # Skip if shorter than 0.1s
                return None

            # FeatureExtractor handles normalization and padding
            inputs = self.feature_extractor(
                y, 
                sampling_rate=self.target_sr, 
                return_tensors="pt", 
                padding=True
            )
            input_values = inputs.input_values.to(self.device)
            
            # 3. Forward Pass
            outputs = self.model(input_values)
            
            # 4. Extract Last Hidden State [1, Seq_Len, Hidden_Dim]
            features = outputs.last_hidden_state.squeeze(0).cpu().numpy()
            
            return features

        except Exception as e:
            logger.debug(f"HuBERT extraction warning for {video_path.name}: {e}")
            return None


# Identity Tracker
class IdentityTracker:
    def __init__(self, smooth_window: int = 5, device: str = 'cuda', conf_thresh: float = 0.5):
        self.smooth_window = smooth_window
        self.device = device
        self.conf_thresh = conf_thresh
        # Load model once
        self.detector = YOLO('yolov8l.pt') 
        self.detector.to(device)

    def track_video(self, video_frames: np.ndarray) -> Optional[Dict[int, List[Tuple]]]:
        trajectories = {}
        
        # Convert 4D numpy array to List of 3D arrays.
        frame_list = [f for f in video_frames]

        results = self.detector.track(
            source=frame_list, 
            persist=True, 
            tracker="bytetrack.yaml",
            classes=[0], 
            conf=self.conf_thresh, 
            verbose=False, 
            device=self.device, 
            stream=True
        )

        for frame_idx, result in enumerate(results):
            if result.boxes is None or result.boxes.id is None: continue
            
            boxes_xywh = result.boxes.xywh.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes_xywh, track_ids):
                # Convert Center-XYWH to TopLeft-XYWH
                x, y, w, h = box[0] - box[2]/2, box[1] - box[3]/2, box[2], box[3]
                
                if track_id not in trajectories: trajectories[track_id] = []
                trajectories[track_id].append((frame_idx, np.array([x, y, w, h])))

        if not trajectories: return None
        return self._smooth_trajectories(trajectories)

    def _smooth_trajectories(self, trajectories: Dict) -> Dict:
        smoothed = {}
        for pid, track in trajectories.items():
            if len(track) < self.smooth_window:
                smoothed[pid] = track
                continue
            frames = [t[0] for t in track]
            bboxes = np.array([t[1] for t in track])
            smoothed_bboxes = np.zeros_like(bboxes)
            # Smooth coordinates
            for i in range(4):
                smoothed_bboxes[:, i] = gaussian_filter1d(bboxes[:, i], sigma=self.smooth_window / 3)
            smoothed[pid] = [(int(f), smoothed_bboxes[idx]) for idx, f in enumerate(frames)]
        return smoothed


# Tube Extractor
class TubeExtractor:
    def __init__(self, crop_size: Tuple[int, int] = (224, 224)):
        self.crop_size = crop_size

    def extract_tubes(self, video_tensor: torch.Tensor, trajectories: Dict, device: str = 'cuda') -> Dict:
        tubes = {}
        video_tensor = video_tensor.to(device)
        _, _, H, W = video_tensor.shape

        for person_id, track in trajectories.items():
            crops = []
            for frame_idx, bbox in track:
                x, y, w, h = map(int, bbox)
                
                # Strict boundary checks
                x, y = max(0, min(x, W-1)), max(0, min(y, H-1))
                w, h = max(1, min(w, W-x)), max(1, min(h, H-y))

                crop = video_tensor[:, frame_idx, y:y+h, x:x+w]
                
                # Interpolate expects [Batch, Channel, H, W]
                crop_resized = F.interpolate(
                    crop.unsqueeze(0), size=self.crop_size, mode='bilinear', align_corners=False
                ).squeeze(0)
                crops.append(crop_resized)

            if crops:
                # Stack temporal dimension
                tubes[person_id] = torch.stack(crops, dim=1).cpu() 

        return tubes


class VGAFDataPreprocessor:
    def __init__(
        self,
        dataset_root: str,
        output_root: str,
        crop_size: Tuple[int, int] = (224, 224),
        target_fps: int = 6,
        device: str = 'cuda'
    ):
        self.dataset_root = Path(dataset_root)
        self.output_root = Path(output_root)
        self.crop_size = crop_size
        self.target_fps = target_fps
        self.device = device
        
        self.data_dir = self.output_root / 'data'
        self.meta_dir = self.output_root / 'metadata'
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        
        self.tracker = IdentityTracker(smooth_window=5, device=device)
        self.tube_extractor = TubeExtractor(crop_size=crop_size)
        
        # --- REPLACED: AudioExtractor with HubertExtractor ---
        self.hubert_extractor = HubertExtractor(device=device)
        
        self.vgaf_to_gavid_map = {1: 0, 2: 1, 3: 2} 
        self.emotion_names = {0: 'Positive', 1: 'Negative', 2: 'Neutral'}

    def get_video_paths(self, split: str) -> List[Path]:
        # Define potential folder names for the split
        if split.lower() == 'train':
            folder_candidates = ['Train', 'train', 'Training', 'training']
        else:
            folder_candidates = ['Val', 'val', 'Validation', 'validation']
            
        # Search locations: specific subdir first, then root
        search_roots = [self.dataset_root / 'VGAF_EmotiW', self.dataset_root]
        
        target_dir = None
        
        # 1. Find the directory
        for root in search_roots:
            if not root.exists(): continue
            for folder_name in folder_candidates:
                candidate_path = root / folder_name
                if candidate_path.exists():
                    target_dir = candidate_path
                    break
            if target_dir: break
            
        if not target_dir:
            logger.warning(f"Could not find video directory for split '{split}'. Checked variants: {folder_candidates}")
            return []
            
        logger.info(f"Found video directory for {split}: {target_dir}")
        
        # 2. Find files
        files = sorted(list(target_dir.glob('*.mp4')))
        if not files:
             logger.warning(f"Directory found ({target_dir}) but contained no .mp4 files.")
             
        return files

    def load_annotations(self, split: str) -> Dict[str, Dict]:
        annotations = {}
        candidates = [
            self.dataset_root / f'{split}_labels.txt',
            self.dataset_root / f'{split.capitalize()}_labels.txt',
            self.dataset_root / 'VGAF_EmotiW' / f'{split}_labels.txt'
        ]
        anno_file = next((p for p in candidates if p.exists()), None)
        if not anno_file: return {}

        with open(anno_file, 'r') as f:
            lines = f.readlines()
        
        start_idx = 1 if 'vid' in lines[0].lower() else 0
        for line in lines[start_idx:]:
            parts = line.strip().split()
            if len(parts) < 2: continue
            vid_name = parts[0].replace('.mp4', '')
            try:
                vgaf_label = int(parts[1])
                if vgaf_label in self.vgaf_to_gavid_map:
                    mapped = self.vgaf_to_gavid_map[vgaf_label]
                    annotations[vid_name] = {
                        'vgaf_label': vgaf_label, 'label': mapped, 
                        'group_emotion': self.emotion_names[mapped]
                    }
            except ValueError: continue
        return annotations

    def load_and_sample_video(self, video_path: Path) -> Optional[np.ndarray]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened(): return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = 30.0 if (fps <= 0 or math.isnan(fps)) else fps
        
        step = max(1, int(round(fps / self.target_fps)))
        frames = []
        count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            if count % step == 0:
                # Validation check for frame dimensions
                if frame.shape[0] > 0 and frame.shape[1] > 0:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            count += 1
            
        cap.release()
        
        if not frames:
            return None
        return np.array(frames)

    def save_to_h5(self, video_id: str, split: str, 
                   full_video_tensor: torch.Tensor, 
                   tubes_data: Dict, 
                   trajectories: Dict,
                   audio_features: Optional[np.ndarray]) -> str:
        
        # Create split-specific folder
        save_dir = self.data_dir / split
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as video_id.h5 inside the split folder
        h5_path = save_dir / f'{video_id}.h5'
        
        # Convert video to uint8 [T,H,W,C]
        full_video_uint8 = (full_video_tensor.permute(1, 2, 3, 0) * 255).byte().numpy()

        with h5py.File(h5_path, 'w') as f:
            # Global Data
            f.create_dataset('full_frames', data=full_video_uint8, compression="lzf", chunks=True)
            
            # Save HuBERT features
            if audio_features is not None:
                f.create_dataset('audio_features', data=audio_features, compression="lzf")
            else:
                # Create empty dataset if no audio features extracted
                f.create_dataset('audio_features', shape=(0,), dtype='f')

            # Person Data
            grp_persons = f.create_group('persons')
            for pid, data in tubes_data.items():
                p_grp = grp_persons.create_group(str(pid))
                visual_uint8 = (data['tube'].permute(1, 2, 3, 0) * 255).byte().numpy()
                p_grp.create_dataset('visual', data=visual_uint8, compression="lzf", chunks=True)
                p_grp.create_dataset('boxes', data=data['box'].numpy())
                raw_coords = np.array([t[1] for t in trajectories[pid]], dtype=np.float32)
                p_grp.create_dataset('raw_coords', data=raw_coords)

        return str(h5_path)

    def reconstruct_metadata(self, h5_path: Path, video_id: str, split: str, annotation: Dict) -> Optional[Dict]:
        """
        Quickly reads existing HDF5 file to reconstruct metadata without reprocessing.
        """
        try:
            with h5py.File(h5_path, 'r') as f:
                num_frames = f['full_frames'].shape[0]
                num_people = len(f['persons'].keys())
                
                # --- MODIFIED CHECK ---
                has_audio = False
                if 'audio_features' in f:
                    has_audio = f['audio_features'].shape[0] > 0
                
            return {
                'video_id': video_id,
                'split': split,
                'num_people': num_people,
                'num_frames': num_frames,
                'has_audio': has_audio,
                **annotation,
                'data_path': str(h5_path)
            }
        except Exception as e:
            logger.warning(f"Failed to read existing file {h5_path}: {e}. Will reprocess.")
            return None

    def process_single_video(self, video_path: Path, annotation: Dict, split: str) -> Optional[Dict]:
        video_id = video_path.stem
        
        # 1. Load Video
        video_frames = self.load_and_sample_video(video_path)
        if video_frames is None: return None
        H, W = video_frames[0].shape[:2]

        # 2. Process Audio (HuBERT)
        audio_features = self.hubert_extractor.extract_features(video_path)

        # 3. Track
        try:
            trajectories = self.tracker.track_video(video_frames)
        except Exception as e:
            logger.error(f"Tracking crashed on {video_id}: {e}")
            return None

        if not trajectories:
            logger.warning(f"No people detected in {video_id}.")
            return None

        # 4. Extract Visual Tubes
        video_tensor = torch.from_numpy(video_frames).permute(3, 0, 1, 2).float().to(self.device) / 255.0
        try:
            tubes_raw = self.tube_extractor.extract_tubes(video_tensor, trajectories, self.device)
            
            # Memory Cleanup
            video_tensor = video_tensor.cpu()
            torch.cuda.empty_cache()
            
            tubes_processed = {}
            for pid, tube in tubes_raw.items():
                track = trajectories[pid]
                norm_boxes = [[bbox[0]/W, bbox[1]/H, bbox[2]/W, bbox[3]/H] for _, bbox in track]
                tubes_processed[pid] = {'tube': tube, 'box': torch.tensor(norm_boxes, dtype=torch.float32)}

            # 5. Save HDF5
            h5_path = self.save_to_h5(video_id, split, video_tensor, tubes_processed, trajectories, audio_features)
            
            metadata = {
                'video_id': video_id,
                'split': split,
                'num_people': len(tubes_processed),
                'num_frames': len(video_frames),
                'has_audio': audio_features is not None,
                **annotation,
                'data_path': h5_path
            }
            return metadata

        except Exception as e:
            logger.error(f"Error processing {video_id}: {str(e)}")
            return None
        finally:
            if 'video_tensor' in locals(): del video_tensor
            torch.cuda.empty_cache()

    def run(self):
        for split in ['train', 'val']:
            logger.info(f"--- Processing Split: {split.upper()} ---")
            annotations = self.load_annotations(split)
            video_paths = self.get_video_paths(split)
            
            all_meta = []
            
            # Create progress bar
            pbar = tqdm(video_paths)
            
            for video_path in pbar:
                video_id = video_path.stem
                if video_id not in annotations:
                    continue
                
                # --- RESUME LOGIC ---
                expected_h5_path = self.data_dir / split / f'{video_id}.h5'
                
                if expected_h5_path.exists():
                    pbar.set_description(f"Skipping {video_id} (Exists)")
                    # Reconstruct metadata from existing file to ensure JSON index is complete
                    meta = self.reconstruct_metadata(expected_h5_path, video_id, split, annotations[video_id])
                    if meta:
                        all_meta.append(meta)
                    else:
                        # If file existed but was corrupt/unreadable, reprocess it
                        meta = self.process_single_video(video_path, annotations[video_id], split)
                        if meta: all_meta.append(meta)
                else:
                    pbar.set_description(f"Processing {video_id}")
                    meta = self.process_single_video(video_path, annotations[video_id], split)
                    if meta: all_meta.append(meta)

            # Write index file (overwrites existing index to ensure it matches current data folder state)
            with open(self.meta_dir / f'{split}_index.json', 'w') as f:
                json.dump(all_meta, f, indent=2)
            
            logger.info(f"Finished {split}. Saved {len(all_meta)} entries to index.")

if __name__ == "__main__":
    DATASET_ROOT = './Datasets/VGAF'
    OUTPUT_ROOT = './VGAF_processed'
    
    processor = VGAFDataPreprocessor(
        dataset_root=DATASET_ROOT,
        output_root=OUTPUT_ROOT,
        crop_size=(224, 224),
        target_fps=6,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print("\n" + "="*50)
    print(" STARTING VGAF PREPROCESSING")
    print("="*50)
    processor.run()