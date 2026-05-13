"""
Perception Test Dataset for SSD-VLM training.
Handles video loading, frame sampling, and memory skill oversampling.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from ssd_vlm.data.video_utils import (
    load_video_frames_dual,
    resolve_video_path,
)

logger = logging.getLogger(__name__)


class PerceptionTestDataset(Dataset):
    """
    Perception Test dataset for multiple-choice video QA.
    
    Features:
    - Efficient video frame loading with caching
    - Uniform frame sampling strategy
    - Memory skill category oversampling (2x default)
    - Preprocessing and caching for slow I/O
    """

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        num_frames: int = 4,
        frame_sampling_strategy: str = "uniform",
        resize_shortest_edge: int = 224,
        memory_skill_oversample_ratio: float = 2.0,
        cache_dir: Optional[str] = None,
        enable_cache: bool = True,
        recent_frames_only: Optional[int] = None,
        chunk_duration: float = 1.0,
        fps: float = 1.0,
    ):
        """
        Initialize Perception Test dataset.
        
        Args:
            data_path: Path to Perception Test root directory
            split: Dataset split (train, val, test)
            num_frames: Number of frames to sample per video
            frame_sampling_strategy: 'uniform' or 'random'
            resize_shortest_edge: Target resolution for frames
            memory_skill_oversample_ratio: Oversample ratio for memory skill (default 2x)
            cache_dir: Directory for frame cache
            enable_cache: Whether to enable caching
        """
        self.data_path = Path(data_path)
        self.split = split
        self.num_frames = num_frames
        self.frame_sampling_strategy = frame_sampling_strategy
        self.resize_shortest_edge = resize_shortest_edge
        self.memory_skill_oversample_ratio = memory_skill_oversample_ratio
        self.enable_cache = enable_cache
        self.recent_frames_only = recent_frames_only or num_frames
        self.chunk_duration = chunk_duration
        self.fps = fps
        
        # Setup cache directory
        if cache_dir is None:
            cache_dir = self.data_path / ".frame_cache"
        self.cache_dir = Path(cache_dir)
        if enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load annotations
        self.annotations_file = self.data_path / f"{split}_annotations.json"
        self.split_file = self.data_path / f"{split}_split.json"
        
        if not self.annotations_file.exists():
            raise FileNotFoundError(f"Annotations not found: {self.annotations_file}")
        
        with open(self.annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        with open(self.split_file, 'r') as f:
            self.split_data = json.load(f)
        
        # Build dataset with oversampling
        self.samples = self._build_samples()
        
        logger.info(f"Loaded {len(self.samples)} samples from {split} split")
    
    # Temporal reference patterns for automatic memory-task detection.
    # Questions with past-tense verbs or temporal markers are classified
    # as memory-relevant, enabling oversampling without answer-level labels.
    _TEMPORAL_PATTERNS = re.compile(
        r'\b(did|was|were|had|before|after|earlier|previously|ago|already'
        r'|happened|occurred|finished|started|began|ended)\b',
        re.IGNORECASE,
    )

    @classmethod
    def _is_memory_by_temporal_reference(cls, question: str) -> bool:
        """Detect memory-relevant questions via temporal reference heuristic."""
        return bool(cls._TEMPORAL_PATTERNS.search(question))

    def _build_samples(self) -> List[Dict[str, Any]]:
        """Build sample list with memory skill oversampling."""
        samples = []
        temporal_oversampled = 0

        for video_id, annotation in self.annotations.items():
            # Check if video exists in split
            if video_id not in self.split_data.get("video_ids", []):
                continue

            # Determine skill category: use annotation if available,
            # otherwise derive automatically from temporal references
            # in the question text (paper Section 3).
            annotated_skill = annotation.get("area", annotation.get("skill", ""))
            question = annotation.get("question", "")
            is_memory = (
                annotated_skill == "memory"
                or self._is_memory_by_temporal_reference(question)
            )

            sample = {
                "video_id": video_id,
                "question": question,
                "options": annotation.get("options", []),
                "answer_idx": annotation.get("answer_id", annotation.get("answer_idx", 0)),
                "skill_category": "memory" if is_memory else annotated_skill,
                "task_type": annotation.get("reasoning", annotation.get("task_type", "")),
            }

            # Add base sample
            samples.append(sample)

            # Apply memory skill oversampling
            if is_memory and self.memory_skill_oversample_ratio > 1.0:
                extra = self.memory_skill_oversample_ratio - 1.0
                num_repeats = int(extra)
                for _ in range(num_repeats):
                    samples.append(sample.copy())
                fractional = extra - num_repeats
                if fractional > 0:
                    # Deterministic fractional oversampling so 1.5x is not rounded to 1.0x.
                    bucket = (sum(ord(ch) for ch in video_id) % 1000) / 1000.0
                    if bucket < fractional:
                        samples.append(sample.copy())
                if annotated_skill != "memory":
                    temporal_oversampled += 1

        logger.info(f"Built {len(samples)} samples with memory oversampling "
                    f"(ratio: {self.memory_skill_oversample_ratio}x, "
                    f"{temporal_oversampled} via temporal heuristic)")
        return samples
    
    def _load_video_dual(self, video_id: str):
        """Decode the video once and return both the preprocessed tensor and raw PIL frames."""
        video_path = resolve_video_path(self.data_path, video_id)
        return load_video_frames_dual(
            video_path=video_path,
            num_frames=self.num_frames,
            tensor_resize_shortest_edge=self.resize_shortest_edge,
            pil_resize_shortest_edge=None,
            frame_sampling_strategy=self.frame_sampling_strategy,
            cache_dir=self.cache_dir,
            enable_cache=self.enable_cache,
            recent_frames_only=(
                self.recent_frames_only
                if self.frame_sampling_strategy in {"recent", "recent_window", "simplestream"}
                else None
            ),
            chunk_duration=self.chunk_duration,
            fps=self.fps,
        )
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.
        
        Returns:
            Dict with keys:
                - frames: [num_frames, 3, H, W] (PyTorch tensor)
                - question: str
                - options: List[str]
                - answer_idx: int
                - skill_category: str
                - task_type: str
                - video_id: str
        """
        sample = self.samples[idx]

        # Decode the video once; return both the preprocessed tensor (lightweight tests)
        # and raw PIL frames (Qwen-VL training/eval).
        frames_tensor, frame_images, frame_indices, total_frames, frame_timestamps, chunk_ids = (
            self._load_video_dual(sample["video_id"])
        )
        
        return {
            "frames": frames_tensor,
            "frame_images": frame_images,
            "question": sample["question"],
            "options": sample["options"],
            "answer_idx": sample["answer_idx"],
            "skill_category": sample["skill_category"],
            "task_type": sample["task_type"],
            "video_id": sample["video_id"],
            "video_relpath": f"videos/{sample['video_id']}.mp4",
            "frame_indices": frame_indices,
            "frame_timestamps": frame_timestamps,
            "chunk_ids": chunk_ids,
            "chunk_duration": self.chunk_duration,
            "fps": self.fps,
            "total_frames": total_frames,
            "source_split": self.split,
            "source_data_path": str(self.data_path.resolve()),
        }


def create_perception_test_dataloader(
    data_path: str,
    split: str = "train",
    batch_size: int = 32,
    num_workers: int = 4,
    num_frames: int = 4,
    memory_skill_oversample_ratio: float = 2.0,
    shuffle: bool = True,
    pin_memory: bool = True,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader for Perception Test dataset.
    
    Args:
        data_path: Path to dataset
        split: Dataset split
        batch_size: Batch size
        num_workers: Number of workers
        num_frames: Frames per video
        memory_skill_oversample_ratio: Memory skill oversampling
        shuffle: Whether to shuffle
        pin_memory: Whether to pin memory
        **kwargs: Additional arguments for PerceptionTestDataset
    
    Returns:
        DataLoader instance
    """
    dataset = PerceptionTestDataset(
        data_path=data_path,
        split=split,
        num_frames=num_frames,
        memory_skill_oversample_ratio=memory_skill_oversample_ratio,
        **kwargs
    )
    
    use_workers = num_workers > 0
    def _collate(batch):
        # Keep this helper usable with PIL frames by leaving them as Python lists.
        out = {}
        for key in batch[0].keys():
            values = [item[key] for item in batch]
            if key == "frames":
                out[key] = torch.stack(values)
            else:
                out[key] = values
        return out

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        drop_last=(split == "train"),
        collate_fn=_collate,
        persistent_workers=True if use_workers else False,
        prefetch_factor=4 if use_workers else None,
    )


if __name__ == "__main__":
    # Test dataset
    logging.basicConfig(level=logging.INFO)
    
    # Create a dummy dataset for testing
    dataset = PerceptionTestDataset(
        data_path="./data/perception_test",
        split="train",
        num_frames=4,
        memory_skill_oversample_ratio=2.0,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get a sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Frames shape: {sample['frames'].shape}")
        print(f"Question: {sample['question']}")
        print(f"Options: {sample['options']}")
