"""Precompute SimpleStream/Qwen3 recent-window PNG frames for OVO-Bench.

This differs from extract_chunk_frames.py: it uses the Qwen-VL video decoder
path that the SimpleStream Qwen3 release uses, then writes a meta.json layout
that OVOBenchDataset can replay without keeping the mp4 chunks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ssd_vlm.data.ovo_bench_dataset import OVOBenchDataset  # noqa: E402
from ssd_vlm.data.video_utils import (  # noqa: E402
    PRECOMPUTED_FRAMES_META,
    _fetch_simplestream_frames,
    resolve_video_path,
)


def _print(message: str) -> None:
    print(message, flush=True)


def _cache_complete(
    cache_dir: Path,
    recent_frames_only: int,
    chunk_duration: float,
    fps: float,
) -> bool:
    meta_path = cache_dir / PRECOMPUTED_FRAMES_META
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if int(meta.get("saved_count", -1)) < int(recent_frames_only):
        return False
    if abs(float(meta.get("extraction_fps", -1.0)) - float(fps)) > 1e-6:
        return False
    if abs(float(meta.get("chunk_duration", -1.0)) - float(chunk_duration)) > 1e-6:
        return False
    pngs = sorted(cache_dir.glob("frame_*.png"))
    return len(pngs) == int(meta.get("saved_count", -1))


def _iter_unique_samples(dataset: OVOBenchDataset, task_type: Optional[str]) -> Iterable[Dict[str, Any]]:
    seen: set[str] = set()
    for sample in dataset.samples:
        if task_type and str(sample.get("task_type")) != task_type:
            continue
        video_id = str(sample["video_id"])
        if video_id in seen:
            continue
        seen.add(video_id)
        yield sample


def _write_one(
    *,
    dataset: OVOBenchDataset,
    sample: Dict[str, Any],
    output_dir: Path,
    recent_frames_only: int,
    chunk_duration: float,
    fps: float,
    overwrite: bool,
) -> Tuple[int, Path]:
    video_path = resolve_video_path(
        data_path=dataset.data_path,
        video_id=sample["video_id"],
        video_relpath=sample.get("video_relpath"),
    )
    cache_dir = output_dir / str(sample["video_id"])
    if overwrite and cache_dir.exists():
        for child in cache_dir.iterdir():
            if child.is_file():
                child.unlink()
    cache_dir.mkdir(parents=True, exist_ok=True)

    previous_cache = os.environ.pop("SSD_VLM_SIMPLESTREAM_FRAME_CACHE_DIR", None)
    try:
        frames, indices, total_frames, timestamps, _chunk_ids = _fetch_simplestream_frames(
            video_path=video_path,
            chunk_duration=chunk_duration,
            fps=fps,
            recent_frames_only=recent_frames_only,
            resize_shortest_edge=None,
        )
    finally:
        if previous_cache is not None:
            os.environ["SSD_VLM_SIMPLESTREAM_FRAME_CACHE_DIR"] = previous_cache

    for stale in cache_dir.glob("frame_*.png"):
        stale.unlink()

    for index, frame in enumerate(frames):
        frame.save(cache_dir / f"frame_{index:02d}.png", format="PNG", optimize=False)

    meta = {
        "total_frames": int(total_frames),
        "source_fps": None,
        "extraction_fps": float(fps),
        "chunk_duration": float(chunk_duration),
        "recent_frames_only": int(recent_frames_only),
        "frame_indices": [int(x) for x in indices],
        "frame_timestamps": [float(x) for x in timestamps],
        "resize_shortest_edge": None,
        "saved_count": len(frames),
        "source_size_bytes": int(video_path.stat().st_size) if video_path.exists() else None,
        "source_video_path": str(video_path),
        "task_type": sample.get("task_type"),
        "source_id": sample.get("source_id"),
    }
    (cache_dir / PRECOMPUTED_FRAMES_META).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return len(frames), video_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-path", default="D:/ssd_video_data")
    parser.add_argument("--anno-path", default="D:/ssd_video_data/ovo_bench_new.json")
    parser.add_argument("--chunked-dir", default="D:/ssd_video_data/chunked_videos")
    parser.add_argument("--output-dir", default="D:/ssd_video_data/chunked_frames")
    parser.add_argument("--task-type", default="", help="Optional single task_type filter.")
    parser.add_argument("--recent-frames-only", type=int, default=4)
    parser.add_argument("--chunk-duration", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--delete-videos-after-cache", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = OVOBenchDataset(
        data_path=args.data_path,
        split="test",
        anno_path=args.anno_path,
        chunked_dir=args.chunked_dir,
        num_frames=args.recent_frames_only,
        recent_frames_only=args.recent_frames_only,
        chunk_duration=args.chunk_duration,
        fps=args.fps,
        use_simplestream_decode=True,
    )
    selected = list(_iter_unique_samples(dataset, args.task_type or None))
    if args.limit > 0:
        selected = selected[: args.limit]

    _print(
        "PRECOMPUTE_START "
        f"videos={len(selected)} task={args.task_type or 'ALL'} output={output_dir} "
        f"recent_frames_only={args.recent_frames_only} delete={args.delete_videos_after_cache}"
    )

    started = time.perf_counter()
    cached = 0
    skipped = 0
    failed = 0
    deleted = 0
    freed_bytes = 0

    for index, sample in enumerate(selected, start=1):
        cache_dir = output_dir / str(sample["video_id"])
        status = "CACHE"
        frame_count = 0
        video_path: Optional[Path] = None
        try:
            if not args.overwrite and _cache_complete(
                cache_dir,
                args.recent_frames_only,
                args.chunk_duration,
                args.fps,
            ):
                skipped += 1
                frame_count = int(json.loads((cache_dir / PRECOMPUTED_FRAMES_META).read_text())["saved_count"])
                video_path = resolve_video_path(
                    data_path=dataset.data_path,
                    video_id=sample["video_id"],
                    video_relpath=sample.get("video_relpath"),
                )
                status = "SKIP"
            else:
                frame_count, video_path = _write_one(
                    dataset=dataset,
                    sample=sample,
                    output_dir=output_dir,
                    recent_frames_only=args.recent_frames_only,
                    chunk_duration=args.chunk_duration,
                    fps=args.fps,
                    overwrite=args.overwrite,
                )
                cached += 1

            if (
                args.delete_videos_after_cache
                and video_path is not None
                and video_path.exists()
                and _cache_complete(cache_dir, args.recent_frames_only, args.chunk_duration, args.fps)
            ):
                size = video_path.stat().st_size
                video_path.unlink()
                deleted += 1
                freed_bytes += size
        except Exception as exc:  # keep long precompute runs resumable
            failed += 1
            status = "FAIL"
            _print(f"ERROR id={sample.get('video_id')} error={exc!r}")

        elapsed = time.perf_counter() - started
        avg = elapsed / max(index, 1)
        eta = avg * (len(selected) - index)
        if index == 1 or index % 25 == 0 or index == len(selected) or status == "FAIL":
            _print(
                "PROGRESS "
                f"{index}/{len(selected)} status={status} id={sample.get('video_id')} "
                f"frames={frame_count} cached={cached} skipped={skipped} failed={failed} "
                f"deleted={deleted} freed_gb={freed_bytes / 1024**3:.2f} "
                f"elapsed_s={elapsed:.1f} eta_s={eta:.1f}"
            )

    _print(
        "PRECOMPUTE_DONE "
        f"videos={len(selected)} cached={cached} skipped={skipped} failed={failed} "
        f"deleted={deleted} freed_gb={freed_bytes / 1024**3:.2f} output={output_dir}"
    )


if __name__ == "__main__":
    main()
