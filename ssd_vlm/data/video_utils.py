"""
Shared video loading and frame sampling utilities.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image as PILImage
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

# cv2 is imported lazily inside read_video_frames / read_video_metadata so
# that callers using only the PNG-based paths (load_precomputed_frames,
# resolve_frame_dir) can import this module without opencv installed.


logger = logging.getLogger(__name__)


_DEFAULT_MEAN = [0.48145466, 0.4578275, 0.40821073]
_DEFAULT_STD = [0.26862954, 0.26130258, 0.27577711]


def build_frame_transform(resize_shortest_edge: int) -> Compose:
    """Build the standard frame preprocessing transform."""
    return Compose([
        Resize((resize_shortest_edge, resize_shortest_edge)),
        ToTensor(),
        Normalize(mean=_DEFAULT_MEAN, std=_DEFAULT_STD),
    ])


def read_video_frames(
    video_path: Path,
    frame_indices: List[int],
) -> Tuple[np.ndarray, int]:
    """Read specific RGB frames from a video by index.

    All callers supply ``frame_indices`` (sampled by the dataset before
    calling here), so there is no full-decode / cache path.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    import cv2  # lazy: only needed for mp4 paths
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Failed to open video: {video_path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            raise ValueError(f"Video has no frames: {video_path}")

        frames = []
        for idx in frame_indices:
            safe_idx = min(max(int(idx), 0), total_frames - 1)
            if safe_idx != int(idx):
                logger.warning(
                    "frame index %s out of bounds for %s (total_frames=%d); clamped to %d",
                    idx, video_path, total_frames, safe_idx,
                )
            cap.set(cv2.CAP_PROP_POS_FRAMES, safe_idx)
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Failed to read frame {safe_idx} from {video_path}")
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return np.asarray(frames), total_frames
    finally:
        cap.release()


def read_video_metadata(video_path: Path) -> Tuple[int, float]:
    """Read frame count and native FPS without decoding video frames."""
    import cv2  # lazy: only needed for mp4 paths
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
    frame_indices: Optional[List[int]] = None,
    recent_frames_only: Optional[int] = None,
    chunk_duration: float = 1.0,
    fps: float = 1.0,
) -> Tuple[List[PILImage.Image], List[int], int, List[float], List[int]]:
    """Load raw RGB PIL frames for Qwen-VL processor input."""
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
    frame_indices: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, List[int], int]:
    """Load, sample, and preprocess frames from a video into a tensor."""
    frame_images, indices, total_frames, _, _ = load_video_frame_images(
        video_path=video_path,
        num_frames=num_frames,
        frame_sampling_strategy=frame_sampling_strategy,
        frame_indices=frame_indices,
    )
    transform = build_frame_transform(resize_shortest_edge)
    return torch.stack([transform(f) for f in frame_images], dim=0), indices, total_frames


def _fetch_simplestream_frames(
    video_path: Path,
    chunk_duration: float,
    fps: float,
    recent_frames_only: int,
    resize_shortest_edge: Optional[int],
) -> Tuple[List[PILImage.Image], List[int], int, List[float], List[int]]:
    """SimpleStream-aligned decode: qwen_vl_utils.fetch_video then chunk-by-time.

    Mirrors lib/recent_window_eval.decode_video_to_chunks_qwen in the
    EvolvingLMMs-Lab/SimpleStream release: decode once at the requested FPS,
    bucket frames into ``chunk_duration``-second windows, keep the last N
    chunks, then return all frames inside them. This is what the published
    Qwen3-VL recent-window scores were computed against.

    When ``chunk_duration * fps == 1.0`` we use the SimpleStream exact-recent
    decoder so only the final N tail frames are read off disk. fetch_video
    otherwise decodes the entire clip at the requested fps and OOMs the CPU
    on long OVO-Bench chunks (1-3 min @ 1080p ~ 6+ GB).
    """
    from qwen_vl_utils.vision_process import fetch_video  # local import; optional dep

    use_exact = abs(float(chunk_duration) * float(fps) - 1.0) < 1e-6
    video_req = {"video": str(video_path), "fps": float(fps)}
    if use_exact:
        from ssd_vlm.data.qwen_exact_recent_decoder import fetch_recent_video_exact

        video, metadata = fetch_recent_video_exact(
            video_req,
            last_nframes=int(recent_frames_only),
            return_video_metadata=True,
        )
    else:
        video, metadata = fetch_video(video_req, return_video_metadata=True)
    if not isinstance(video, torch.Tensor) or video.ndim != 4:
        raise ValueError(f"Unexpected qwen_vl_utils output for video={video_path!r}")

    meta = metadata if isinstance(metadata, dict) else {}
    raw_fps = max(float(meta.get("fps", fps if fps > 0 else 1.0)), 1e-6)
    frame_indices_meta = meta.get("frames_indices")
    if isinstance(frame_indices_meta, torch.Tensor):
        frame_indices_meta = frame_indices_meta.detach().cpu().reshape(-1).tolist()
    elif frame_indices_meta is not None and not isinstance(frame_indices_meta, (list, tuple)):
        try:
            frame_indices_meta = list(frame_indices_meta)
        except TypeError:
            frame_indices_meta = None
    if frame_indices_meta is None or len(frame_indices_meta) != int(video.shape[0]):
        frame_indices_meta = list(range(int(video.shape[0])))
    frame_indices_meta = [int(x) for x in frame_indices_meta]

    # Bucket frames by chunk index (timestamp // chunk_duration).
    chunk = max(float(chunk_duration), 1e-6)
    buckets: dict[int, List[Tuple[PILImage.Image, float, int]]] = {}
    for i, idx in enumerate(frame_indices_meta):
        ts = float(idx) / raw_fps
        chunk_idx = int(ts // chunk)
        arr = video[i].clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        image = PILImage.fromarray(arr)
        image = _resize_pil_shortest_edge(image, resize_shortest_edge)
        buckets.setdefault(chunk_idx, []).append((image, ts, idx))
    del video

    # Keep the last N chunks (by chunk_idx).
    sorted_chunk_ids = sorted(buckets)
    window_size = max(1, int(recent_frames_only))
    recent_ids = sorted_chunk_ids[-window_size:]

    pil_frames: List[PILImage.Image] = []
    timestamps: List[float] = []
    selected_indices: List[int] = []
    chunk_ids: List[int] = []
    for cid in recent_ids:
        for image, ts, idx in buckets[cid]:
            pil_frames.append(image)
            timestamps.append(ts)
            selected_indices.append(idx)
            chunk_ids.append(cid)

    total_frames = int(len(frame_indices_meta))
    return pil_frames, selected_indices, total_frames, timestamps, chunk_ids


PRECOMPUTED_FRAMES_META = "meta.json"


def load_precomputed_frames(
    frame_dir: Path,
    num_frames: int,
    *,
    expected_fps: float = 1.0,
    expected_chunk_duration: float = 1.0,
    resize_shortest_edge: Optional[int] = None,
) -> Tuple[List[PILImage.Image], List[int], int, List[float], List[int]]:
    """Load N most-recent PNG frames previously produced by extract_chunk_frames.py.

    Layout under ``frame_dir`` is:
      ``frame_<NN>.png`` (sorted lexicographically == saved order ==
        oldest-to-newest within the recent window)
      ``meta.json`` describing the sampling parameters used at extract time.

    Raises ``ValueError`` if the saved sampling profile does not match
    ``expected_fps``/``expected_chunk_duration`` — refusing to silently
    return frames sampled under a different policy.
    """
    meta_path = frame_dir / PRECOMPUTED_FRAMES_META
    if not meta_path.exists():
        raise FileNotFoundError(f"Precomputed-frame meta not found: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    saved_fps = float(meta.get("extraction_fps", -1.0))
    saved_cd = float(meta.get("chunk_duration", -1.0))
    if abs(saved_fps - float(expected_fps)) > 1e-6 or abs(saved_cd - float(expected_chunk_duration)) > 1e-6:
        raise ValueError(
            f"Precomputed frames at {frame_dir} were extracted with "
            f"fps={saved_fps}, chunk_duration={saved_cd}; eval requests "
            f"fps={expected_fps}, chunk_duration={expected_chunk_duration}. "
            f"Re-extract or fix the config."
        )

    saved_indices: List[int] = [int(i) for i in meta.get("frame_indices", [])]
    saved_timestamps: List[float] = [float(t) for t in meta.get("frame_timestamps", [])]
    total_frames = int(meta.get("total_frames", len(saved_indices)))
    saved_count = int(meta.get("saved_count", len(saved_indices)))

    pngs = sorted(frame_dir.glob("frame_*.png"))
    if len(pngs) != saved_count:
        raise ValueError(
            f"Precomputed frames at {frame_dir}: meta says saved_count={saved_count} "
            f"but found {len(pngs)} png files"
        )
    if len(pngs) != len(saved_indices) or len(pngs) != len(saved_timestamps):
        raise ValueError(
            f"Precomputed frames at {frame_dir}: indices/timestamps length mismatch "
            f"({len(saved_indices)}/{len(saved_timestamps)} vs {len(pngs)} pngs)"
        )

    take = min(int(num_frames), len(pngs))
    selected_pngs = pngs[-take:]
    selected_indices = saved_indices[-take:]
    selected_timestamps = saved_timestamps[-take:]

    chunk = max(float(expected_chunk_duration), 1e-6)
    chunk_ids = [int(ts // chunk) for ts in selected_timestamps]

    pil_frames: List[PILImage.Image] = []
    for path in selected_pngs:
        image = PILImage.open(path).convert("RGB")
        image.load()
        pil_frames.append(_resize_pil_shortest_edge(image, resize_shortest_edge))

    return pil_frames, selected_indices, total_frames, selected_timestamps, chunk_ids


def load_video_frames_dual(
    video_path: Optional[Path],
    num_frames: int,
    *,
    tensor_resize_shortest_edge: int = 224,
    pil_resize_shortest_edge: Optional[int] = None,
    frame_sampling_strategy: str = "uniform",
    frame_indices: Optional[List[int]] = None,
    recent_frames_only: Optional[int] = None,
    chunk_duration: float = 1.0,
    fps: float = 1.0,
    use_simplestream_decode: bool = False,
    precomputed_frame_dir: Optional[Path] = None,
) -> Tuple[torch.Tensor, List[PILImage.Image], List[int], int, List[float], List[int]]:
    """
    One-pass loader returning both preprocessed tensor frames and raw PIL frames.

    The dataset classes previously called ``load_video_frame_images`` and
    ``load_video_frames`` back-to-back per item; that decoded each video twice
    (or hit the frame cache twice). This helper samples and decodes once.

    When ``use_simplestream_decode`` is True, switches to the SimpleStream
    decode path (qwen_vl_utils.fetch_video at the requested FPS, bucket by
    chunk_duration seconds, keep last ``recent_frames_only`` chunks). This is
    the path required to reproduce the published SimpleStream Qwen3-VL 4f
    numbers.

    When ``precomputed_frame_dir`` is set, skip mp4 decoding entirely and read
    the last N PNG frames previously written by
    ``scripts/extract_chunk_frames.py``. ``video_path`` is then ignored and
    may be ``None``.

    Returns: (tensor, pil_frames, indices, total_frames, timestamps, chunk_ids).
    """
    if precomputed_frame_dir is not None:
        pil_frames, indices, total_frames, timestamps, chunk_ids = load_precomputed_frames(
            frame_dir=precomputed_frame_dir,
            num_frames=int(recent_frames_only or num_frames),
            expected_fps=fps,
            expected_chunk_duration=chunk_duration,
            resize_shortest_edge=pil_resize_shortest_edge,
        )
    elif use_simplestream_decode:
        pil_frames, indices, total_frames, timestamps, chunk_ids = _fetch_simplestream_frames(
            video_path=video_path,
            chunk_duration=chunk_duration,
            fps=fps,
            recent_frames_only=int(recent_frames_only or num_frames),
            resize_shortest_edge=pil_resize_shortest_edge,
        )
    else:
        pil_frames, indices, total_frames, timestamps, chunk_ids = load_video_frame_images(
            video_path=video_path,
            num_frames=num_frames,
            frame_sampling_strategy=frame_sampling_strategy,
            resize_shortest_edge=pil_resize_shortest_edge,
            frame_indices=frame_indices,
            recent_frames_only=recent_frames_only,
            chunk_duration=chunk_duration,
            fps=fps,
        )
    transform = build_frame_transform(tensor_resize_shortest_edge)
    tensor = torch.stack([transform(frame) for frame in pil_frames], dim=0)
    return tensor, pil_frames, indices, total_frames, timestamps, chunk_ids


def resolve_frame_dir(
    data_path: Path,
    video_id: str,
    chunked_frames_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Return ``<chunked_frames>/<video_id>/`` if it has a meta.json, else None.

    Used by datasets to opt into the precomputed-frames path when the
    extractor has been run; callers fall back to ``resolve_video_path`` if
    this returns ``None``.
    """
    base = Path(chunked_frames_dir) if chunked_frames_dir else (Path(data_path) / "chunked_frames")
    candidate = base / str(video_id)
    if (candidate / PRECOMPUTED_FRAMES_META).exists():
        return candidate
    return None


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
