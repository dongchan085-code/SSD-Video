"""
Shared video loading and frame sampling utilities.
"""

import hashlib
import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image as PILImage
from torchvision.transforms import Compose, Normalize, Resize, ToTensor


_DEFAULT_MEAN = [0.48145466, 0.4578275, 0.40821073]
_DEFAULT_STD = [0.26862954, 0.26130258, 0.27577711]


def build_frame_transform(resize_shortest_edge: int) -> Compose:
    """Build the standard frame preprocessing transform."""
    return Compose([
        Resize((resize_shortest_edge, resize_shortest_edge)),
        ToTensor(),
        Normalize(mean=_DEFAULT_MEAN, std=_DEFAULT_STD),
    ])


def _cache_key(video_path: Path) -> str:
    digest = hashlib.sha1(str(video_path.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{video_path.stem}_{digest}"


def read_video_frames(
    video_path: Path,
    cache_dir: Optional[Path] = None,
    enable_cache: bool = True,
) -> Tuple[np.ndarray, int]:
    """Read all RGB frames from a video, optionally using a compressed cache."""
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cache_path = None
    if enable_cache and cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = _cache_key(video_path)
        npy_path = cache_dir / f"{key}_frames.npy"
        meta_path = cache_dir / f"{key}_meta.json"
        npz_path = cache_dir / f"{key}_frames.npz"

        # Prefer fast npy memmap cache
        if npy_path.exists() and meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            frames = np.load(npy_path, mmap_mode="r")
            return frames, int(meta["total_frames"])

        # Fallback: legacy npz cache (read once, re-save as npy)
        if npz_path.exists():
            cached = np.load(npz_path)
            frames_np = cached["frames"]
            total = int(cached["total_frames"])
            np.save(npy_path, frames_np)
            with open(meta_path, "w") as f:
                json.dump({"total_frames": total}, f)
            return frames_np, total

        cache_path = npy_path

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        raise ValueError(f"Video has no frames: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    frames_np = np.asarray(frames)
    if enable_cache and cache_path is not None:
        np.save(cache_path, frames_np)
        meta_out = cache_path.with_name(
            cache_path.name.replace("_frames.npy", "_meta.json")
        )
        with open(meta_out, "w") as f:
            json.dump({"total_frames": total_frames}, f)

    return frames_np, total_frames


def sample_frame_indices(
    total_frames: int,
    num_frames: int,
    strategy: str = "uniform",
) -> np.ndarray:
    """Sample frame indices according to the requested strategy."""
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")

    if strategy == "uniform":
        return np.linspace(0, total_frames - 1, num_frames, dtype=int)

    if strategy == "random":
        replace = total_frames < num_frames
        indices = np.random.choice(total_frames, num_frames, replace=replace)
        return np.sort(indices.astype(int))

    raise ValueError(f"Unknown sampling strategy: {strategy}")


def load_video_frames(
    video_path: Path,
    num_frames: int,
    frame_sampling_strategy: str = "uniform",
    resize_shortest_edge: int = 224,
    cache_dir: Optional[Path] = None,
    enable_cache: bool = True,
    frame_indices: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, List[int], int]:
    """
    Load, sample, and preprocess frames from a video.
    """
    frames, total_frames = read_video_frames(
        video_path=video_path,
        cache_dir=cache_dir,
        enable_cache=enable_cache,
    )
    if frame_indices:
        indices = np.asarray(
            [min(max(int(idx), 0), len(frames) - 1) for idx in frame_indices],
            dtype=int,
        )
    else:
        indices = sample_frame_indices(
            total_frames=len(frames),
            num_frames=num_frames,
            strategy=frame_sampling_strategy,
        )
    transform = build_frame_transform(resize_shortest_edge)
    sampled_frames = []
    for frame in frames[indices]:
        sampled_frames.append(transform(PILImage.fromarray(frame.astype(np.uint8))))

    return torch.stack(sampled_frames, dim=0), indices.tolist(), total_frames


def resolve_video_path(data_path: Path, video_id: str, video_relpath: Optional[str] = None) -> Path:
    """Resolve a video path from a dataset root plus optional relative path."""
    candidates = []
    if video_relpath:
        candidates.append(data_path / video_relpath)
    candidates.append(data_path / "videos" / f"{video_id}.mp4")

    for path in candidates:
        if path.exists():
            return path

    videos_dir = data_path / "videos"
    if videos_dir.exists():
        for match in videos_dir.glob(f"{video_id}.*"):
            if match.is_file():
                return match

    raise FileNotFoundError(f"Could not resolve video path for {video_id} under {data_path}")
