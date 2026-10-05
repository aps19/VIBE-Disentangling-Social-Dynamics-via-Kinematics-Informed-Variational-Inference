"""
GAVID -> VIBE primitives in a single pass.

For VGAF/GECV the pipeline is two steps (preprocessing_*.py writes raw frames,
tubes and HuBERT features to an intermediate HDF5, then extract_primitives.py
encodes them). For GAVID both steps run per video, so the large intermediate
frame dump is never written to disk. The models and settings are the same:

    6 Hz frame sampling -> YOLOv8 + ByteTrack -> Gaussian-smoothed trajectories
    HuBERT (audio), VideoMAE (global), DINOv2 (env + agent tubes), RoBERTa (text)

Output layout (matches dataloading/vibe_dataset.py):

    <output_root>/<split>/<video_id>_csync.h5
        global_v_seq      [1, 1568, 768]   VideoMAE tokens (16 frames)
        env_feat          [1, 768]         DINOv2 CLS of the middle frame
        physics_sync      [K, K]           velocity-synchrony matrix
        audio_seq         [T_a, 768]       HuBERT last hidden state
        text_anch         [1, 768]         RoBERTa pooler output of Description
        person_sequences/p_i [T_i, 768]    DINOv2 CLS per agent crop
        attrs: label (1=Positive, 2=Neutral, 3=Negative), group_emotion, ...

Usage:
    python data_preprocessing/extract_gavid.py \
        --gavid-root /media/abhishek/disk3/GAVID_Extension/dataset/GAVID \
        --output-root ./GAVID_VIBE_PRIMITIVES
"""

import os
import re
import sys
import math
import logging
import argparse
import warnings
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing_gecv import HubertExtractor, IdentityTracker
from extract_primitives import VIBEFeatureEngine

from ultralytics import YOLO

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# VIBEDataset maps {1: 0, 2: 1, 3: 2} and test_model names {0: Positive, 1: Neutral, 2: Negative}
LABEL_MAP = {'positive': 1, 'neutral': 2, 'negative': 3}
META_COLS = ['group_emotion', 'specific_emotion', 'emotion_intensity', 'interaction_type', 'action_cues']


class GAVIDTracker(IdentityTracker):
    """IdentityTracker with configurable weights and a tracker reset per video.

    The base class tracks with persist=True (needed so ByteTrack keeps state
    across the frames of a numpy-list source), but never resets, so track
    state leaks from one video into the next.
    """

    def __init__(self, weights, smooth_window=5, device='cuda', conf_thresh=0.5):
        self.smooth_window = smooth_window
        self.device = device
        self.conf_thresh = conf_thresh
        self.detector = YOLO(weights)
        self.detector.to(device)

    def track_video(self, video_frames):
        predictor = getattr(self.detector, 'predictor', None)
        if predictor is not None and hasattr(predictor, 'trackers'):
            for tracker in predictor.trackers:
                tracker.reset()
        return super().track_video(video_frames)


class GAVIDExtractor:
    def __init__(self, yolo_weights, max_k=8, target_fps=6, crop_size=224,
                 dino_batch=32, device='cuda'):
        self.max_k = max_k
        self.target_fps = target_fps
        self.crop_size = crop_size
        self.dino_batch = dino_batch
        self.device = device

        self.tracker = GAVIDTracker(yolo_weights, smooth_window=5, device=device)
        self.hubert = HubertExtractor(device=device)
        self.engine = VIBEFeatureEngine(device=device)

    def load_and_sample_video(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = 30.0 if (fps <= 0 or math.isnan(fps)) else fps
        step = max(1, int(round(fps / self.target_fps)))

        frames = []
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if count % step == 0 and frame.shape[0] > 0 and frame.shape[1] > 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            count += 1
        cap.release()
        return np.array(frames) if frames else None

    def select_agents(self, trajectories):
        """Keep the max_k agents with the longest tracks (most screen time)."""
        if not trajectories:
            return {}
        ranked = sorted(trajectories.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        return dict(ranked[:self.max_k])

    def crop_tubes(self, frames, agents):
        """Returns one [T_i, crop, crop, 3] uint8 tube per agent, in agent order."""
        _, H, W, _ = frames.shape
        tubes = []
        for track in agents.values():
            crops = []
            for frame_idx, bbox in track:
                x, y, w, h = map(int, bbox)
                x, y = max(0, min(x, W - 1)), max(0, min(y, H - 1))
                w, h = max(1, min(w, W - x)), max(1, min(h, H - y))
                crop = frames[frame_idx, y:y + h, x:x + w]
                crops.append(cv2.resize(crop, (self.crop_size, self.crop_size),
                                        interpolation=cv2.INTER_LINEAR))
            tubes.append(np.stack(crops))
        return tubes

    @torch.no_grad()
    def encode(self, frames, tubes, agents, description):
        eng = self.engine
        H, W = frames.shape[1:3]

        with torch.autocast(device_type='cuda', enabled=self.device.startswith('cuda')):
            # 1. Global scene sequence (VideoMAE, 16 uniformly spaced frames)
            idx = np.linspace(0, len(frames) - 1, 16).astype(int)
            v_in = eng.vmae_processor([frames[i] for i in idx], return_tensors="pt").to(self.device)
            global_v_seq = eng.vmae_model(**v_in).last_hidden_state.float().cpu().numpy()

            # 2. Environmental prior (DINOv2 CLS of the middle frame)
            env_in = eng.dino_processor(frames[len(frames) // 2], return_tensors="pt").to(self.device)
            env_feat = eng.dino_model(**env_in).last_hidden_state[:, 0, :].float().cpu().numpy()

            # 3. Local agent sequences (DINOv2 CLS per crop, batched over all agents)
            local_p_seqs = []
            if tubes:
                all_crops = [c for tube in tubes for c in tube]
                feats = []
                for i in range(0, len(all_crops), self.dino_batch):
                    p_in = eng.dino_processor(all_crops[i:i + self.dino_batch], return_tensors="pt").to(self.device)
                    feats.append(eng.dino_model(**p_in).last_hidden_state[:, 0, :].float().cpu())
                feats = torch.cat(feats).numpy()
                cursor = 0
                for tube in tubes:
                    local_p_seqs.append(feats[cursor:cursor + len(tube)])
                    cursor += len(tube)

            # 4. Text anchor (RoBERTa pooler)
            t_in = eng.text_tokenizer(description, return_tensors="pt", padding=True,
                                      truncation=True).to(self.device)
            text_anch = eng.text_model(**t_in).pooler_output.float().cpu().numpy()

        # 5. Physics synchrony over the selected agents
        coords = {pid: np.array([b for _, b in track], dtype=np.float32) for pid, track in agents.items()}
        physics_sync = eng.calculate_physics_sync(coords, (H, W), max_k=self.max_k)

        return {
            'global_v_seq': global_v_seq,
            'env_feat': env_feat,
            'local_p_seqs': local_p_seqs,
            'physics_sync': physics_sync.astype(np.float32),
            'text_anch': text_anch,
        }

    def process_video(self, video_path, row, out_path):
        frames = self.load_and_sample_video(video_path)
        if frames is None:
            raise RuntimeError("could not decode video")

        audio = self.hubert.extract_features(video_path)
        has_audio = audio is not None
        if not has_audio:
            audio = np.zeros((1, 768), dtype=np.float32)

        trajectories = self.tracker.track_video(frames) or {}
        agents = self.select_agents(trajectories)
        tubes = self.crop_tubes(frames, agents)

        description = '' if pd.isna(row.get('Description')) else str(row['Description'])
        prims = self.encode(frames, tubes, agents, description)

        tmp_path = out_path.with_suffix('.h5.tmp')
        with h5py.File(tmp_path, 'w') as f:
            f.create_dataset('global_v_seq', data=prims['global_v_seq'], compression='gzip')
            f.create_dataset('env_feat', data=prims['env_feat'], compression='gzip')
            f.create_dataset('physics_sync', data=prims['physics_sync'], compression='gzip')
            f.create_dataset('audio_seq', data=audio.astype(np.float32), compression='gzip')
            f.create_dataset('text_anch', data=prims['text_anch'], compression='gzip')

            p_group = f.create_group('person_sequences')
            for i, p_seq in enumerate(prims['local_p_seqs']):
                p_group.create_dataset(f'p_{i}', data=p_seq, compression='gzip')

            f.attrs['label'] = row['label']
            f.attrs['video_id'] = row['video_key']
            f.attrs['description'] = description
            for col in META_COLS:
                f.attrs[col] = '' if pd.isna(row.get(col)) else str(row[col])
            f.attrs['num_frames'] = len(frames)
            f.attrs['num_tracks'] = len(trajectories)
            f.attrs['num_agents'] = len(agents)
            f.attrs['has_audio'] = has_audio
        os.replace(tmp_path, out_path)

        return {'num_frames': len(frames), 'num_tracks': len(trajectories),
                'num_agents': len(agents), 'has_audio': has_audio}


def load_labels(xlsx_path):
    df = pd.read_excel(xlsx_path)
    df['video_key'] = df['video_id'].astype(str).str.strip().apply(
        lambda v: re.sub(r'\.mp4$', '', v, flags=re.IGNORECASE))
    df['label'] = df['group_emotion'].astype(str).str.strip().str.lower().map(LABEL_MAP)

    bad = df['label'].isna()
    if bad.any():
        logger.warning(f"{xlsx_path.name}: dropping {bad.sum()} rows with unknown group_emotion "
                       f"{sorted(df.loc[bad, 'group_emotion'].astype(str).unique())}")
        df = df[~bad]
    df['label'] = df['label'].astype(int)

    dup = df['video_key'].duplicated()
    if dup.any():
        logger.warning(f"{xlsx_path.name}: dropping {dup.sum()} duplicate video_id rows")
        df = df[~dup]
    return df.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Extract VIBE primitives for the GAVID dataset")
    parser.add_argument('--gavid-root', type=Path, required=True,
                        help="Folder holding train/ val/ test/ video folders and the label .xlsx files")
    parser.add_argument('--output-root', type=Path, default=Path('./GAVID_VIBE_PRIMITIVES'))
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    parser.add_argument('--train-xlsx', default='train.xlsx',
                        help="Label file for the train split, relative to --gavid-root (e.g. train_new.xlsx)")
    parser.add_argument('--yolo-weights', default='yolov8l.pt',
                        help="Path to YOLOv8 weights (ultralytics downloads yolov8l.pt if missing)")
    parser.add_argument('--max-k', type=int, default=8, help="Max agents kept per clip")
    parser.add_argument('--target-fps', type=int, default=6)
    parser.add_argument('--suffix', default='_csync.h5',
                        help="Output file suffix; train_model.py/test_model.py glob '*_csync.h5'")
    parser.add_argument('--limit', type=int, default=None, help="Process only the first N videos per split")
    parser.add_argument('--overwrite', action='store_true', help="Re-extract files that already exist")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    extractor = GAVIDExtractor(args.yolo_weights, max_k=args.max_k, target_fps=args.target_fps,
                               device=args.device)

    for split in args.splits:
        xlsx_name = args.train_xlsx if split == 'train' else f'{split}.xlsx'
        xlsx_path = args.gavid_root / xlsx_name
        video_dir = args.gavid_root / split
        if not xlsx_path.exists() or not video_dir.exists():
            logger.warning(f"Skipping {split}: missing {xlsx_path} or {video_dir}")
            continue

        df = load_labels(xlsx_path)
        if args.limit:
            df = df.head(args.limit)
        dest_dir = args.output_root / split
        dest_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"--- {split.upper()}: {len(df)} labelled videos from {xlsx_name} ---")

        records = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"GAVID {split}"):
            key = row['video_key']
            video_path = video_dir / f'{key}.mp4'
            out_path = dest_dir / f'{key}{args.suffix}'
            rec = {'video_id': key, 'group_emotion': row['group_emotion'], 'label': row['label']}

            if not video_path.exists():
                rec['status'] = 'missing_video'
            elif out_path.exists() and not args.overwrite:
                rec['status'] = 'exists'
            else:
                try:
                    rec.update(extractor.process_video(video_path, row, out_path))
                    rec['status'] = 'ok'
                except Exception as e:
                    logger.error(f"Failed {split}/{key}: {e}")
                    rec['status'] = f'error: {e}'
                finally:
                    if args.device.startswith('cuda'):
                        torch.cuda.empty_cache()
            records.append(rec)

        meta = pd.DataFrame(records)
        meta.to_csv(args.output_root / f'{split}_index.csv', index=False)
        counts = meta['status'].str.split(':').str[0].value_counts().to_dict()
        logger.info(f"Finished {split}: {counts}")


if __name__ == '__main__':
    main()
