"""
OVO-Bench dataset utilities for real video-backed evaluation.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from torch.utils.data import Dataset

from ssd_vlm.data.video_utils import load_video_frames, resolve_video_path

logger = logging.getLogger(__name__)


LOCK_TASKS = {"OCR", "ATR", "OJR", "STU", "ACR", "FPD"}
FORK_TASKS = {"EPM", "ASI", "HLD"}


class OVOBenchDataset(Dataset):
    """OVO-Bench dataset with on-demand frame loading."""

    def __init__(
        self,
        data_path: str,
        split: str = "test",
        num_frames: int = 4,
        frame_sampling_strategy: str = "uniform",
        resize_shortest_edge: int = 224,
        cache_dir: Optional[str] = None,
        enable_cache: bool = True,
    ):
        self.data_path = Path(data_path)
        self.split = split
        self.num_frames = num_frames
        self.frame_sampling_strategy = frame_sampling_strategy
        self.resize_shortest_edge = resize_shortest_edge
        self.enable_cache = enable_cache
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_path / ".frame_cache"
        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        split_file = self.data_path / f"{split}_split.json"
        annotations_file = self.data_path / f"{split}_annotations.json"
        if not split_file.exists() or not annotations_file.exists():
            raise FileNotFoundError(f"OVO-Bench data not found in {self.data_path}")

        with open(split_file, "r") as f:
            split_data = json.load(f)
        with open(annotations_file, "r") as f:
            annotations = json.load(f)

        self.samples = []
        for video_id in split_data.get("video_ids", []):
            if video_id not in annotations:
                continue
            annotation = annotations[video_id]
            self.samples.append({
                "video_id": video_id,
                "question": annotation.get("question", ""),
                "options": annotation.get("options", []),
                "answer_idx": annotation.get("answer_idx", 0),
                "task_type": annotation.get("task_type", ""),
                "video_relpath": annotation.get("video_relpath", f"videos/{video_id}.mp4"),
            })

        logger.info(f"Loaded {len(self.samples)} OVO-Bench {split} samples from {self.data_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx].copy()
        video_path = resolve_video_path(
            data_path=self.data_path,
            video_id=sample["video_id"],
            video_relpath=sample.get("video_relpath"),
        )
        frames, frame_indices, total_frames = load_video_frames(
            video_path=video_path,
            num_frames=self.num_frames,
            frame_sampling_strategy=self.frame_sampling_strategy,
            resize_shortest_edge=self.resize_shortest_edge,
            cache_dir=self.cache_dir,
            enable_cache=self.enable_cache,
        )
        sample.update({
            "frames": frames,
            "frame_indices": frame_indices,
            "total_frames": total_frames,
            "video_path": str(video_path),
        })
        return sample
