"""
Perception Test Dataset for SSD-VLM training.
Handles video loading, frame sampling, and memory skill oversampling.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

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
        
        # Image transforms
        self.transforms = Compose([
            Resize((resize_shortest_edge, resize_shortest_edge)),
            ToTensor(),
            Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
        
        logger.info(f"Loaded {len(self.samples)} samples from {split} split")
    
    def _build_samples(self) -> List[Dict[str, Any]]:
        """Build sample list with memory skill oversampling."""
        samples = []
        
        for video_id, annotation in self.annotations.items():
            # Check if video exists in split
            if video_id not in self.split_data.get("video_ids", []):
                continue
            
            sample = {
                "video_id": video_id,
                "question": annotation.get("question", ""),
                "options": annotation.get("options", []),
                "answer_idx": annotation.get("answer_id", annotation.get("answer_idx", 0)),
                "skill_category": annotation.get("area", annotation.get("skill", "")),
                "task_type": annotation.get("reasoning", annotation.get("task_type", "")),
            }
            
            # Add base sample
            samples.append(sample)
            
            # Apply memory skill oversampling
            if (annotation.get("area", annotation.get("skill", "")) == "memory" and 
                self.memory_skill_oversample_ratio > 1.0):
                num_repeats = int(self.memory_skill_oversample_ratio - 1)
                for _ in range(num_repeats):
                    samples.append(sample.copy())
        
        logger.info(f"Built {len(samples)} samples with memory oversampling "
                   f"(ratio: {self.memory_skill_oversample_ratio}x)")
        return samples
    
    def _get_video_frames(self, video_id: str) -> np.ndarray:
        """
        Load and sample frames from video.
        
        Returns:
            np.ndarray: [num_frames, H, W, 3]
        """
        video_path = self.data_path / "videos" / f"{video_id}.mp4"
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Try to load from cache first
        cache_path = self.cache_dir / f"{video_id}_frames.npz"
        if self.enable_cache and cache_path.exists():
            frames_data = np.load(cache_path)
            frames = frames_data["frames"]
            total_frames = frames_data["total_frames"]
        else:
            # Load video
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise IOError(f"Failed to open video: {video_path}")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                raise ValueError(f"Video has no frames: {video_path}")
            
            # Read all frames
            all_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                all_frames.append(frame)
            cap.release()
            
            frames = np.array(all_frames)
            
            # Cache frames
            if self.enable_cache:
                np.savez_compressed(
                    cache_path,
                    frames=frames,
                    total_frames=total_frames
                )
        
        # Sample frames
        if self.frame_sampling_strategy == "uniform":
            indices = np.linspace(0, len(frames) - 1, self.num_frames, dtype=int)
        elif self.frame_sampling_strategy == "random":
            indices = np.random.choice(len(frames), self.num_frames, replace=False)
            indices = np.sort(indices)
        else:
            raise ValueError(f"Unknown sampling strategy: {self.frame_sampling_strategy}")
        
        sampled_frames = frames[indices]
        return sampled_frames
    
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
        
        # Load frames
        frames = self._get_video_frames(sample["video_id"])
        
        # Apply transforms to each frame
        # torchvision transforms expect PIL Images, not numpy arrays
        from PIL import Image as PILImage
        processed_frames = []
        for frame in frames:
            pil_frame = PILImage.fromarray(frame.astype(np.uint8))
            frame_tensor = self.transforms(pil_frame)
            processed_frames.append(frame_tensor)
        
        frames_tensor = torch.stack(processed_frames, dim=0)
        
        return {
            "frames": frames_tensor,
            "question": sample["question"],
            "options": sample["options"],
            "answer_idx": sample["answer_idx"],
            "skill_category": sample["skill_category"],
            "task_type": sample["task_type"],
            "video_id": sample["video_id"],
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
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        drop_last=(split == "train"),
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
