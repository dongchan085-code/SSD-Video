"""Cache SimpleStream recent-window frames for OVO-Bench chunk videos.

The cache stores the final N SimpleStream frames as PNG files so selected
chunk videos can be removed without changing evaluation inputs.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from ssd_vlm.data.ovo_bench_dataset import OVOBenchDataset
from ssd_vlm.data.video_utils import _fetch_simplestream_frames, resolve_video_path


def _print(message: str) -> None:
    print(message, flush=True)


def _sample_video_path(dataset: OVOBenchDataset, sample: Dict[str, Any]) -> Path:
    return resolve_video_path(
        data_path=dataset.data_path,
        video_id=sample["video_id"],
        video_relpath=sample.get("video_relpath"),
    )


def _select_samples(dataset: OVOBenchDataset, task_type: str | None) -> List[Dict[str, Any]]:
    if not task_type:
        return list(dataset.samples)
    return [
        sample for sample in dataset.samples
        if str(sample.get("task_type")) == task_type
    ]


def _cache_complete(
    cache_dir: Path,
    recent_frames_only: int,
    chunk_duration: float,
    fps: float,
) -> bool:
    metadata_path = cache_dir / "metadata.json"
    if not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if int(metadata.get("recent_frames_only", 0)) < int(recent_frames_only):
        return False
    if abs(float(metadata.get("chunk_duration", -1.0)) - float(chunk_duration)) > 1e-6:
        return False
    if abs(float(metadata.get("fps", -1.0)) - float(fps)) > 1e-6:
        return False
    frames = metadata.get("frames") or []
    return bool(frames) and all((cache_dir / str(item.get("file"))).exists() for item in frames)


def _iter_unique_samples(
    dataset: OVOBenchDataset,
    samples: Iterable[Dict[str, Any]],
) -> Iterable[tuple[Dict[str, Any], Path]]:
    seen: set[str] = set()
    for sample in samples:
        video_path = _sample_video_path(dataset, sample)
        key = str(video_path)
        if key in seen:
            continue
        seen.add(key)
        yield sample, video_path


def _write_frame_cache(
    sample: Dict[str, Any],
    video_path: Path,
    cache_dir: Path,
    recent_frames_only: int,
    chunk_duration: float,
    fps: float,
    overwrite: bool,
) -> int:
    if overwrite and cache_dir.exists():
        for child in cache_dir.iterdir():
            if child.is_file():
                child.unlink()
    cache_dir.mkdir(parents=True, exist_ok=True)

    previous_cache = os.environ.pop("SSD_VLM_SIMPLESTREAM_FRAME_CACHE_DIR", None)
    try:
        frames, indices, total_frames, timestamps, chunk_ids = _fetch_simplestream_frames(
            video_path=video_path,
            chunk_duration=chunk_duration,
            fps=fps,
            recent_frames_only=recent_frames_only,
            resize_shortest_edge=None,
        )
    finally:
        if previous_cache is not None:
            os.environ["SSD_VLM_SIMPLESTREAM_FRAME_CACHE_DIR"] = previous_cache

    records = []
    for index, frame in enumerate(frames):
        filename = f"{index:04d}.png"
        frame.save(cache_dir / filename, format="PNG")
        records.append({
            "file": filename,
            "frame_index": int(indices[index]),
            "timestamp": float(timestamps[index]),
            "chunk_id": int(chunk_ids[index]),
        })

    metadata = {
        "video_id": sample.get("video_id"),
        "source_id": sample.get("source_id"),
        "task_type": sample.get("task_type"),
        "source_video_path": str(video_path),
        "recent_frames_only": int(recent_frames_only),
        "chunk_duration": float(chunk_duration),
        "fps": float(fps),
        "total_frames": int(total_frames),
        "frames": records,
    }
    (cache_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="D:/ssd_video_data")
    parser.add_argument("--anno-path", default="D:/ssd_video_data/ovo_bench_full.json")
    parser.add_argument("--chunked-dir", default="D:/ssd_video_data/chunked_videos")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-type", default="HLD", help="Use an empty value to cache all tasks.")
    parser.add_argument("--all-tasks", action="store_true")
    parser.add_argument("--recent-frames-only", type=int, default=8)
    parser.add_argument("--chunk-duration", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
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
    task_type = None if args.all_tasks else (args.task_type or None)
    selected_samples = _select_samples(dataset, task_type)
    if args.limit is not None:
        selected_samples = selected_samples[: args.limit]
    unique_items = list(_iter_unique_samples(dataset, selected_samples))
    total = len(unique_items)

    _print(
        "CACHE_START "
        f"task={task_type or 'ALL'} videos={total} output={output_dir} "
        f"recent_frames_only={args.recent_frames_only} delete={args.delete_videos_after_cache}"
    )

    start = time.perf_counter()
    cached = 0
    skipped = 0
    deleted = 0
    bytes_deleted = 0
    for index, (sample, video_path) in enumerate(unique_items, start=1):
        cache_dir = output_dir / video_path.stem
        if not args.overwrite and _cache_complete(
            cache_dir,
            args.recent_frames_only,
            args.chunk_duration,
            args.fps,
        ):
            frame_count = len(json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))["frames"])
            skipped += 1
            status = "SKIP"
        else:
            frame_count = _write_frame_cache(
                sample=sample,
                video_path=video_path,
                cache_dir=cache_dir,
                recent_frames_only=args.recent_frames_only,
                chunk_duration=args.chunk_duration,
                fps=args.fps,
                overwrite=args.overwrite,
            )
            cached += 1
            status = "CACHE"

        if args.delete_videos_after_cache and _cache_complete(
            cache_dir,
            args.recent_frames_only,
            args.chunk_duration,
            args.fps,
        ) and video_path.exists():
            size = video_path.stat().st_size
            video_path.unlink()
            deleted += 1
            bytes_deleted += size

        elapsed = time.perf_counter() - start
        avg = elapsed / index
        eta = avg * (total - index)
        _print(
            "PROGRESS "
            f"{index}/{total} status={status} id={sample.get('video_id')} "
            f"frames={frame_count} deleted={deleted} "
            f"freed_gb={bytes_deleted / 1024**3:.2f} "
            f"elapsed_s={elapsed:.1f} eta_s={eta:.1f}"
        )

    _print(
        "CACHE_DONE "
        f"videos={total} cached={cached} skipped={skipped} deleted={deleted} "
        f"freed_gb={bytes_deleted / 1024**3:.2f} output={output_dir}"
    )


if __name__ == "__main__":
    main()
