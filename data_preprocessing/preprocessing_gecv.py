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

            # 2. Process inputs
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
            # e.g., [1, T, 768] for base model
            features = outputs.last_hidden_state.squeeze(0).cpu().numpy()
            
            return features

        except Exception as e:
            logger.debug(f"HuBERT extraction warning for {video_path.name}: {e}")
            return None


class IdentityTracker:
    def __init__(self, smooth_window: int = 5, device: str = 'cuda', conf_thresh: float = 0.5):
        self.smooth_window = smooth_window
        self.device = device
        self.conf_thresh = conf_thresh
        self.detector = YOLO('yolov8l.pt') 
        self.detector.to(device)

    def track_video(self, video_frames: np.ndarray) -> Optional[Dict[int, List[Tuple]]]:
        trajectories = {}
        frame_list = [f for f in video_frames] # Fix for YOLO dimension bug
        results = self.detector.track(
            source=frame_list, persist=True, tracker="bytetrack.yaml",
            classes=[0], conf=self.conf_thresh, verbose=False, device=self.device, stream=True
        )
        for frame_idx, result in enumerate(results):
            if result.boxes is None or result.boxes.id is None: continue
            boxes_xywh = result.boxes.xywh.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy().astype(int)
            for box, track_id in zip(boxes_xywh, track_ids):
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
            for i in range(4):
                smoothed_bboxes[:, i] = gaussian_filter1d(bboxes[:, i], sigma=self.smooth_window / 3)
            smoothed[pid] = [(int(f), smoothed_bboxes[idx]) for idx, f in enumerate(frames)]
        return smoothed

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
                x, y = max(0, min(x, W-1)), max(0, min(y, H-1))
                w, h = max(1, min(w, W-x)), max(1, min(h, H-y))
                crop = video_tensor[:, frame_idx, y:y+h, x:x+w]
                crop_resized = F.interpolate(
                    crop.unsqueeze(0), size=self.crop_size, mode='bilinear', align_corners=False
                ).squeeze(0)
                crops.append(crop_resized)
            if crops:
                tubes[person_id] = torch.stack(crops, dim=1).cpu() 
        return tubes


class GEVCDataPreprocessor:
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
        
        # Directories
        self.data_dir = self.output_root / 'data'
        self.meta_dir = self.output_root / 'metadata'
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        
        # Init Components
        self.tracker = IdentityTracker(smooth_window=5, device=device)
        self.tube_extractor = TubeExtractor(crop_size=crop_size)
        
        # AudioExtractor with HubertExtractor ---
        self.hubert_extractor = HubertExtractor(device=device)
        
        # Mapping (Adjust based on your specific GAVID labels)
        self.emotion_map = {
            'Positive': 0, 'Negative': 1, 'Neutral': 2,
            0: 0, 1: 1, 2: 2
        }

    def get_video_paths(self, split: str) -> List[Path]:
        # --- ROBUST PATH FINDING ---
        if split.lower() == 'train':
            folder_candidates = ['Train', 'train', 'Training', 'training']
        elif split.lower() == 'test':
            folder_candidates = ['Test', 'test', 'Testing', 'testing']
        else:
            folder_candidates = ['Val', 'val', 'Validation', 'validation']
            
        search_roots = [self.dataset_root / 'GAVID', self.dataset_root]
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
        
        # 2. Find files (Recursive + Case Insensitive)
        files = [
            p for p in target_dir.rglob('*') 
            if p.is_file() and p.suffix.lower() == '.mp4'
        ]
        
        return sorted(files)

    def load_annotations(self, split: str) -> Dict[str, Dict]:
        if split == 'test':
            logger.info("Test split selected: processing without labels.")
            return {}

        annotations = {}
        
        # Search for CSV and Excel files in root and Annotations subdir
        # Added support for .xlsx files
        patterns = [f'*{split}*.csv', f'*{split}*.xlsx']
        candidates = []
        for pat in patterns:
            candidates.extend(list(self.dataset_root.glob(pat)))
            candidates.extend(list((self.dataset_root / 'Annotations').glob(pat)))
        
        # Prioritize files with 'label' in the name (e.g. train_labels.xlsx over train.xlsx)
        candidates.sort(key=lambda p: 'label' not in p.name.lower())

        if not candidates:
            logger.warning(f"No annotation file found for {split}. Processing without labels.")
            return {}

        # Try to read candidates until we find a valid one
        valid_df = None
        valid_path = None
        
        for cand in candidates:
            try:
                if cand.suffix.lower() == '.csv':
                    df = pd.read_csv(cand)
                elif cand.suffix.lower() == '.xlsx':
                    df = pd.read_excel(cand)
                else:
                    continue
                
                # Normalize columns: lower case and strip whitespace
                df.columns = [str(c).lower().strip() for c in df.columns]
                
                # Check for 'video_id' or similar column to confirm it's the right file
                # We look for 'video' AND 'id', or 'filename', or 'file_name'
                vid_col = next((c for c in df.columns if ('video' in c and 'id' in c) or 'filename' in c or 'file_name' in c), None)
                
                if vid_col:
                    valid_df = df
                    valid_path = cand
                    break
            except Exception as e:
                logger.debug(f"Failed to read candidate {cand}: {e}")
                continue

        if valid_df is None:
            logger.warning(f"Found annotation files but could not parse any for {split}. Processing without labels.")
            return {}

        logger.info(f"Loading annotations from: {valid_path}")
        df = valid_df
        
        # Identify columns
        vid_col = next((c for c in df.columns if ('video' in c and 'id' in c) or 'filename' in c or 'file_name' in c), None)
        emo_col = next((c for c in df.columns if 'group_emotion' in c or 'label' in c or 'emotion' in c), None)
        desc_col = next((c for c in df.columns if 'description' in c or 'caption' in c), None)

        if not vid_col:
            logger.error(f"Could not find video ID column. Columns: {df.columns}")
            return {}

        for _, row in df.iterrows():
            try:
                # Parse Video ID
                raw_vid_id = str(row[vid_col])
                # Remove extension if present
                key = re.sub(r'\.mp4$', '', raw_vid_id, flags=re.IGNORECASE).strip()
                
                label_data = {
                    'description': str(row[desc_col]) if desc_col and not pd.isna(row[desc_col]) else ""
                }

                # Parse Emotion Label
                if emo_col and not pd.isna(row[emo_col]):
                    raw_emo = str(row[emo_col]).strip()
                    # Try to map string label or use integer directly
                    if raw_emo.isdigit():
                        label_int = int(raw_emo)
                    else:
                        label_int = self.emotion_map.get(raw_emo, -1)
                    
                    label_data['label'] = label_int
                    label_data['group_emotion'] = raw_emo
                else:
                    label_data['label'] = -1
                    label_data['group_emotion'] = 'Unknown'
                
                annotations[key] = label_data
                
            except Exception as e:
                logger.debug(f"Row parse error: {e}")
                continue
                
        logger.info(f"Loaded {len(annotations)} annotations.")
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
                if frame.shape[0] > 0 and frame.shape[1] > 0:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            count += 1
            
        cap.release()
        return np.array(frames) if frames else None

    def save_to_h5(self, filename_stem: str, split: str, 
                   full_video_tensor: torch.Tensor, 
                   tubes_data: Dict, 
                   trajectories: Dict,
                   audio_features: Optional[np.ndarray]) -> str:
        
        save_dir = self.data_dir / split
        save_dir.mkdir(parents=True, exist_ok=True)
        
        h5_path = save_dir / f'{filename_stem}.h5'
        
        full_video_uint8 = (full_video_tensor.permute(1, 2, 3, 0) * 255).byte().numpy()

        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('full_frames', data=full_video_uint8, compression="lzf", chunks=True)
            
            # Save HuBERT features instead of spectrogram ---
            if audio_features is not None:
                f.create_dataset('audio_features', data=audio_features, compression="lzf")
            else:
                f.create_dataset('audio_features', shape=(0,), dtype='f')

            grp_persons = f.create_group('persons')
            for pid, data in tubes_data.items():
                p_grp = grp_persons.create_group(str(pid))
                visual_uint8 = (data['tube'].permute(1, 2, 3, 0) * 255).byte().numpy()
                p_grp.create_dataset('visual', data=visual_uint8, compression="lzf", chunks=True)
                p_grp.create_dataset('boxes', data=data['box'].numpy())
                raw_coords = np.array([t[1] for t in trajectories[pid]], dtype=np.float32)
                p_grp.create_dataset('raw_coords', data=raw_coords)

        return str(h5_path)

    def reconstruct_metadata(self, h5_path: Path, video_name: str, split: str, annotation: Dict) -> Optional[Dict]:
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
                'video_name': video_name,
                'split': split,
                'num_people': num_people,
                'num_frames': num_frames,
                'has_audio': has_audio,
                'label': annotation.get('label', -1),
                'group_emotion': annotation.get('group_emotion', 'Unknown'),
                'description': annotation.get('description', ''),
                'data_path': str(h5_path)
            }
        except Exception as e:
            logger.warning(f"Failed to read existing file {h5_path}: {e}. Will reprocess.")
            return None

    def process_single_video(self, video_path: Path, annotation: Dict, split: str) -> Optional[Dict]:
        vid_stem = video_path.stem 
        
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
            logger.error(f"Tracking error {vid_stem}: {e}")
            return None

        if not trajectories:
            logger.warning(f"No people detected in {vid_stem}.")
            return None

        # 4. Extract Visual Tubes
        video_tensor = torch.from_numpy(video_frames).permute(3, 0, 1, 2).float().to(self.device) / 255.0
        try:
            tubes_raw = self.tube_extractor.extract_tubes(video_tensor, trajectories, self.device)
            video_tensor = video_tensor.cpu()
            torch.cuda.empty_cache()
            
            tubes_processed = {}
            for pid, tube in tubes_raw.items():
                track = trajectories[pid]
                norm_boxes = [[bbox[0]/W, bbox[1]/H, bbox[2]/W, bbox[3]/H] for _, bbox in track]
                tubes_processed[pid] = {'tube': tube, 'box': torch.tensor(norm_boxes, dtype=torch.float32)}

            # 5. Save HDF5
            h5_path = self.save_to_h5(vid_stem, split, video_tensor, tubes_processed, trajectories, audio_features)
            
            metadata = {
                'video_name': vid_stem,
                'split': split,
                'num_people': len(tubes_processed),
                'num_frames': len(video_frames),
                'has_audio': audio_features is not None,
                'label': annotation.get('label', -1),
                'group_emotion': annotation.get('group_emotion', 'Unknown'),
                'description': annotation.get('description', ''),
                'data_path': h5_path
            }
            return metadata

        except Exception as e:
            logger.error(f"Error processing {vid_stem}: {str(e)}")
            return None
        finally:
            if 'video_tensor' in locals(): del video_tensor
            torch.cuda.empty_cache()

    def run(self):
        for split in ['train', 'val', 'test']:
            logger.info(f"--- Processing Split: {split.upper()} ---")
            
            annotations = self.load_annotations(split)
            video_paths = self.get_video_paths(split)
            
            if not video_paths:
                logger.warning(f"No videos found for {split}. Skipping.")
                continue

            all_meta = []
            
            # Use TQDM with resume logic
            pbar = tqdm(video_paths)
            
            for video_path in pbar:
                key = video_path.stem 
                anno = annotations.get(key, {}) 
                
                # Only process if we have annotations OR it is the test set
                if anno or split == 'test':
                    
                    # --- RESUME LOGIC ---
                    expected_h5_path = self.data_dir / split / f'{key}.h5'
                    
                    if expected_h5_path.exists():
                        pbar.set_description(f"Skipping {key} (Exists)")
                        meta = self.reconstruct_metadata(expected_h5_path, key, split, anno)
                        if meta:
                            all_meta.append(meta)
                        else:
                            # File corrupt or unreadable, re-process
                            meta = self.process_single_video(video_path, anno, split)
                            if meta: all_meta.append(meta)
                    else:
                        pbar.set_description(f"Processing {key}")
                        meta = self.process_single_video(video_path, anno, split)
                        if meta: all_meta.append(meta)

            # Dump JSON index
            with open(self.meta_dir / f'{split}_index.json', 'w') as f:
                json.dump(all_meta, f, indent=2)
                
            logger.info(f"Finished {split}. Saved {len(all_meta)} entries to index.")


if __name__ == "__main__":
    # UPDATE THESE PATHS
    DATASET_ROOT = './GEVC'
    OUTPUT_ROOT = './GEVC_processed'
    
    processor = GEVCDataPreprocessor(
        dataset_root=DATASET_ROOT,
        output_root=OUTPUT_ROOT,
        crop_size=(224, 224),
        target_fps=6,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print("\n" + "="*50)
    print(" STARTING GEVC PREPROCESSING")
    print("="*50)
    processor.run()
    print("\nProcessing Complete.")