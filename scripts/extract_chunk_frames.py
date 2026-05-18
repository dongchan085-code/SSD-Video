"""Pre-extract OVO-Bench chunk frames as PNG and delete the source mp4.

Why this exists
---------------
``chunked_videos/*.mp4`` totals ~100 GB on the 176 GB D:\\ Azure VM disk,
but every production eval reads at most 4 frames per chunk (and the
sweep ablations cap at 32). Saving the recent-window frames as PNGs at
short-edge 384 reduces the per-chunk footprint from ~33 MB to ~1-2 MB
while preserving the exact frames the eval would have decoded.

The script reuses ``ssd_vlm.data.video_utils.sample_recent_window_indices``
(the same index sampler the dataset calls at eval-time) so the saved
frames are byte-identical to what ``OVOBenchDataset`` would have loaded.

Output layout
-------------
    <output_dir>/<video_id>/frame_00.png   # oldest in the recent window
    <output_dir>/<video_id>/frame_01.png
    ...
    <output_dir>/<video_id>/meta.json      # sampling profile + indices

``meta.json`` records ``extraction_fps``, ``chunk_duration``,
``frame_indices`` and ``frame_timestamps`` so the loader
(``ssd_vlm.data.video_utils.load_precomputed_frames``) can refuse to
serve frames sampled under a different policy.

Usage
-----
    python scripts/extract_chunk_frames.py \\
        --input_dir D:/ssd_video_data/chunked_videos \\
        --output_dir D:/ssd_video_data/chunked_frames \\
        --recent_frames 32 \\
        --fps 1.0 \\
        --chunk_duration 1.0 \\
        --resize_shortest_edge 384 \\
        --delete_source

Rerunning is safe: chunks whose ``meta.json`` already matches the
saved-png count are skipped, so a crashed run can resume in place.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

# Ensure ssd_vlm is importable when this script is invoked from a checkout
# without `pip install -e .`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
from PIL import Image as PILImage  # noqa: E402

from ssd_vlm.data.video_utils import (  # noqa: E402
    PRECOMPUTED_FRAMES_META,
    _resize_pil_shortest_edge,
    read_video_frames,
    read_video_metadata,
    sample_recent_window_indices,
)


logger = logging.getLogger("extract_chunk_frames")


def _already_extracted(out_dir: Path) -> bool:
    meta_path = out_dir / PRECOMPUTED_FRAMES_META
    if not meta_path.exists():
        return False
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        saved = int(meta.get("saved_count", -1))
    except (OSError, json.JSONDecodeError):
        return False
    pngs = list(out_dir.glob("frame_*.png"))
    return saved >= 0 and saved == len(pngs)


def _extract_one(
    video_path: Path,
    out_dir: Path,
    recent_frames: int,
    fps: float,
    chunk_duration: float,
    resize_shortest_edge: Optional[int],
) -> int:
    total_frames, source_fps = read_video_metadata(video_path)
    indices_arr = sample_recent_window_indices(
        total_frames=total_frames,
        recent_frames_only=int(recent_frames),
        fps=float(fps),
        source_fps=float(source_fps),
    )
    indices: List[int] = [int(x) for x in indices_arr.tolist()]
    if not indices:
        raise ValueError(f"sample_recent_window_indices produced no indices for {video_path}")

    frames_np, _ = read_video_frames(
        video_path=video_path,
        frame_indices=indices,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    # Wipe any stale frame_*.png from a prior interrupted run so the
    # saved_count matches what we write here.
    for stale in out_dir.glob("frame_*.png"):
        stale.unlink()

    timestamp_fps = source_fps if source_fps and source_fps > 0 else (fps if fps and fps > 0 else 1.0)
    timestamps = [float(idx) / timestamp_fps for idx in indices]

    for i, frame_arr in enumerate(frames_np):
        image = PILImage.fromarray(np.asarray(frame_arr, dtype=np.uint8))
        image = _resize_pil_shortest_edge(image, resize_shortest_edge)
        image.save(out_dir / f"frame_{i:02d}.png", format="PNG", optimize=False)

    meta = {
        "total_frames": int(total_frames),
        "source_fps": float(source_fps) if source_fps else None,
        "extraction_fps": float(fps),
        "chunk_duration": float(chunk_duration),
        "recent_frames_only": int(recent_frames),
        "frame_indices": indices,
        "frame_timestamps": timestamps,
        "resize_shortest_edge": int(resize_shortest_edge) if resize_shortest_edge else None,
        "saved_count": len(indices),
        "source_size_bytes": int(video_path.stat().st_size),
    }
    with open(out_dir / PRECOMPUTED_FRAMES_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return len(indices)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input_dir", required=True, help="chunked_videos/ directory")
    parser.add_argument("--output_dir", required=True, help="destination chunked_frames/ directory")
    parser.add_argument("--recent_frames", type=int, default=32, help="frames kept per chunk (the last N)")
    parser.add_argument("--fps", type=float, default=1.0, help="streaming FPS the eval uses")
    parser.add_argument("--chunk_duration", type=float, default=1.0, help="chunk_duration seconds for meta.json")
    parser.add_argument("--resize_shortest_edge", type=int, default=384, help="PNG short edge; 0 to keep native")
    parser.add_argument(
        "--delete_source",
        action="store_true",
        help="delete the .mp4 after a successful extraction (default: keep)",
    )
    parser.add_argument("--glob", default="*.mp4", help="filename pattern under input_dir")
    parser.add_argument("--limit", type=int, default=0, help="stop after N chunks (0 = no limit)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.exists():
        raise SystemExit(f"input_dir not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    resize = args.resize_shortest_edge if args.resize_shortest_edge and args.resize_shortest_edge > 0 else None

    videos = sorted(input_dir.glob(args.glob))
    if args.limit > 0:
        videos = videos[: args.limit]
    logger.info("Found %d candidate videos under %s", len(videos), input_dir)

    extracted = 0
    skipped = 0
    failed = 0
    bytes_freed = 0
    started = time.monotonic()

    for i, video_path in enumerate(videos, 1):
        video_id = video_path.stem
        out_dir = output_dir / video_id

        if _already_extracted(out_dir):
            skipped += 1
            if args.delete_source and video_path.exists():
                bytes_freed += video_path.stat().st_size
                video_path.unlink()
            continue

        try:
            saved = _extract_one(
                video_path=video_path,
                out_dir=out_dir,
                recent_frames=args.recent_frames,
                fps=args.fps,
                chunk_duration=args.chunk_duration,
                resize_shortest_edge=resize,
            )
        except Exception as exc:
            failed += 1
            logger.error("extract failed for %s: %s", video_path, exc)
            continue

        extracted += 1
        if args.delete_source:
            bytes_freed += video_path.stat().st_size
            video_path.unlink()

        if i % 50 == 0 or i == len(videos):
            dt = max(time.monotonic() - started, 1e-6)
            logger.info(
                "[%d/%d] last=%s saved=%d extracted=%d skipped=%d failed=%d freed=%.1fGB rate=%.1f/s",
                i, len(videos), video_id, saved, extracted, skipped, failed,
                bytes_freed / (1024 ** 3), i / dt,
            )

    logger.info(
        "Done. extracted=%d skipped=%d failed=%d bytes_freed=%.2fGB",
        extracted, skipped, failed, bytes_freed / (1024 ** 3),
    )


if __name__ == "__main__":
    main()
