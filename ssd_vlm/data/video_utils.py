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
    frame_indices: Optional[List[int]] = None,
) -> Tuple[np.ndarray, int]:
    """Read all RGB frames from a video, optionally using a compressed cache."""
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        raise ValueError(f"Video has no frames: {video_path}")

    if frame_indices is not None:
        frames = []
        for idx in frame_indices:
            safe_idx = min(max(int(idx), 0), total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, safe_idx)
            ret, frame = cap.read()
            if not ret:
                cap.release()
                raise ValueError(f"Failed to read frame {safe_idx} from {video_path}")
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return np.asarray(frames), total_frames

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


def read_video_metadata(video_path: Path) -> Tuple[int, float]:
    """Read frame count and native FPS without decoding video frames."""
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Failed to open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    if total_frames <= 0:
        raise ValueError(f"Video has no frames: {video_path}")
    return total_frames, source_fps


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


def sample_recent_window_indices(
    total_frames: int,
    recent_frames_only: int,
    fps: float = 1.0,
    source_fps: Optional[float] = None,
) -> np.ndarray:
    """Sample the last N observed frames, matching SimpleStream's recency window."""
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if recent_frames_only <= 0:
        raise ValueError("recent_frames_only must be positive")

    # If native FPS is known, first emulate observing at the requested streaming FPS,
    # then keep only the most recent N observed frames.
    if source_fps and source_fps > 0 and fps and fps > 0:
        stride = max(1, int(round(source_fps / fps)))
        observed = np.arange(0, total_frames, stride, dtype=int)
        if observed.size == 0:
            observed = np.asarray([total_frames - 1], dtype=int)
        return observed[-recent_frames_only:]

    start = max(0, total_frames - recent_frames_only)
    return np.arange(start, total_frames, dtype=int)


def _resize_pil_shortest_edge(image: PILImage.Image, shortest_edge: Optional[int]) -> PILImage.Image:
    if not shortest_edge:
        return image
    width, height = image.size
    if min(width, height) == shortest_edge:
        return image
    scale = shortest_edge / float(min(width, height))
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return image.resize(new_size, PILImage.BICUBIC)


def load_video_frame_images(
    video_path: Path,
    num_frames: int,
    frame_sampling_strategy: str = "uniform",
    resize_shortest_edge: Optional[int] = None,
    cache_dir: Optional[Path] = None,
    enable_cache: bool = True,
    frame_indices: Optional[List[int]] = None,
    recent_frames_only: Optional[int] = None,
    chunk_duration: float = 1.0,
    fps: float = 1.0,
) -> Tuple[List[PILImage.Image], List[int], int, List[float], List[int]]:
    """
    Load raw RGB PIL frames for Qwen-VL processor input.

    The default path preserves the old uniform sampler. Setting
    frame_sampling_strategy="recent" or recent_frames_only uses a SimpleStream-style
    recent window: observe frames at the requested streaming FPS, keep the last N.
    """
    total_frames, source_fps = read_video_metadata(video_path)

    if frame_indices:
        indices = np.asarray(
            [min(max(int(idx), 0), total_frames - 1) for idx in frame_indices],
            dtype=int,
        )
    elif frame_sampling_strategy in {"recent", "recent_window", "simplestream"} or recent_frames_only:
        indices = sample_recent_window_indices(
            total_frames=total_frames,
            recent_frames_only=int(recent_frames_only or num_frames),
            fps=fps,
            source_fps=source_fps,
        )
    else:
        indices = sample_frame_indices(
            total_frames=total_frames,
            num_frames=num_frames,
            strategy=frame_sampling_strategy,
        )

    frames, total_frames = read_video_frames(
        video_path=video_path,
        cache_dir=cache_dir,
        enable_cache=enable_cache,
        frame_indices=indices.tolist(),
    )

    pil_frames: List[PILImage.Image] = []
    for frame in frames:
        image = PILImage.fromarray(frame.astype(np.uint8))
        pil_frames.append(_resize_pil_shortest_edge(image, resize_shortest_edge))

    timestamp_fps = source_fps if source_fps and source_fps > 0 else (fps if fps and fps > 0 else 1.0)
    timestamps = [float(idx) / timestamp_fps for idx in indices.tolist()]
    chunk = max(float(chunk_duration), 1e-6)
    chunk_ids = [int(ts // chunk) for ts in timestamps]
    return pil_frames, indices.tolist(), total_frames, timestamps, chunk_ids


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
    frame_images, indices, total_frames, _, _ = load_video_frame_images(
        video_path=video_path,
        num_frames=num_frames,
        frame_sampling_strategy=frame_sampling_strategy,
        resize_shortest_edge=None,
        cache_dir=cache_dir,
        enable_cache=enable_cache,
        frame_indices=frame_indices,
    )
    transform = build_frame_transform(resize_shortest_edge)
    sampled_frames = [transform(frame) for frame in frame_images]

    return torch.stack(sampled_frames, dim=0), indices, total_frames


def resolve_video_path(data_path: Path, video_id: str, video_relpath: Optional[str] = None) -> Path:
    """Resolve a video path from a dataset root plus optional relative path."""
    candidates = []
    if video_relpath:
        candidates.append(data_path / video_relpath)
    candidates.append(data_path / "chunked_videos" / f"{video_id}.mp4")
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
