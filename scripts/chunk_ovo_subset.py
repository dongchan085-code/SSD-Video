"""Create SimpleStream-style OVO chunk videos from source videos.

Uses ffmpeg stream-copy (`-c copy`) by default — no re-encoding, ~10x faster
than the previous OpenCV-based path. Falls back to OpenCV re-encode if
stream-copy fails or if `--reencode` is passed. Keyframe alignment means the
copied clip may extend by up to one GOP past the requested end_time; for the
4-frame OVO eval that samples recent frames at 1 fps this is acceptable.
"""

import argparse
import json
import math
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import imageio_ffmpeg
from tqdm import tqdm


FORWARD_TASKS = {"REC", "SSR", "CRR"}
FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()


def source_name(annotation: Dict[str, Any]) -> str:
    value = annotation.get("video") or annotation.get("video_path")
    if value:
        return str(value).replace("\\", "/")
    return f"{annotation.get('id')}.mp4"


def requested_chunks(annotation: Dict[str, Any]) -> Iterable[Tuple[str, float]]:
    video_id = str(annotation["id"])
    task = annotation.get("task", annotation.get("task_type", ""))
    if task in FORWARD_TASKS:
        for idx, item in enumerate(annotation.get("test_info") or []):
            realtime = item.get("realtime", annotation.get("realtime", 0))
            yield f"{video_id}_{idx}.mp4", float(realtime)
    else:
        yield f"{video_id}.mp4", float(annotation.get("realtime", 0))


def _probe_duration_seconds(src_path: Path) -> float:
    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return total / fps if fps > 0 else 0.0


def _verify_chunk(dst_path: Path) -> bool:
    if not dst_path.exists() or dst_path.stat().st_size == 0:
        return False
    cap = cv2.VideoCapture(str(dst_path))
    ok = cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) > 0
    cap.release()
    return ok


def _ffmpeg_stream_copy(src_path: Path, dst_path: Path, end_time: float) -> bool:
    cmd = [
        FFMPEG_EXE, "-hide_banner", "-loglevel", "error", "-y",
        "-ss", "0",
        "-to", f"{max(end_time, 0.0):.3f}",
        "-i", str(src_path),
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        "-movflags", "+faststart",
        str(dst_path),
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        return False
    return _verify_chunk(dst_path)


def _opencv_reencode(src_path: Path, dst_path: Path, end_time: float) -> bool:
    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source video: {src_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    end_frame = int(math.ceil(max(end_time, 0.0) * fps))
    if end_frame <= 0:
        end_frame = total_frames or int(fps)
    if total_frames > 0:
        end_frame = min(end_frame, total_frames)

    writer = cv2.VideoWriter(str(dst_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create chunk video: {dst_path}")
    written = 0
    try:
        while written < end_frame:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            written += 1
    finally:
        writer.release()
        cap.release()
    if written == 0:
        raise RuntimeError(f"No frames written for {dst_path} from {src_path}")
    return True


def chunk_video(src_path: Path, dst_path: Path, end_time: float, overwrite: bool, reencode: bool = False) -> bool:
    if dst_path.exists() and dst_path.stat().st_size > 0 and not overwrite:
        if _verify_chunk(dst_path):
            return False
        dst_path.unlink()
    if dst_path.exists() and dst_path.stat().st_size == 0:
        dst_path.unlink()

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if not reencode:
        if _ffmpeg_stream_copy(src_path, dst_path, end_time):
            return True
        if dst_path.exists():
            dst_path.unlink()
        print(f"[chunk] ffmpeg copy failed for {dst_path.name}, falling back to re-encode", flush=True)

    return _opencv_reencode(src_path, dst_path, end_time)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk an OVO-Bench annotation subset.")
    parser.add_argument("--anno_path", required=True)
    parser.add_argument("--src_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow_missing", action="store_true")
    parser.add_argument("--reencode", action="store_true",
                        help="Force OpenCV re-encode instead of ffmpeg stream copy")
    args = parser.parse_args()

    annotations: List[Dict[str, Any]] = json.loads(Path(args.anno_path).read_text(encoding="utf-8"))
    src_dir = Path(args.src_dir)
    output_dir = Path(args.output_dir)

    jobs = []
    missing = []
    for annotation in annotations:
        src_path = src_dir.joinpath(*Path(source_name(annotation)).parts)
        if not src_path.exists():
            missing.append(str(src_path))
            continue
        for chunk_name, end_time in requested_chunks(annotation):
            jobs.append((src_path, output_dir / chunk_name, end_time))

    if missing and not args.allow_missing:
        raise SystemExit(f"Missing {len(missing)} source videos, first few: {missing[:10]}")

    created = 0
    skipped = 0
    for src_path, dst_path, end_time in tqdm(jobs, desc="Chunking OVO subset", unit="chunk"):
        did_create = chunk_video(src_path, dst_path, end_time, args.overwrite, reencode=args.reencode)
        created += int(did_create)
        skipped += int(not did_create)

    print(f"Created {created} chunks, skipped {skipped} existing chunks in {output_dir}")


if __name__ == "__main__":
    main()
