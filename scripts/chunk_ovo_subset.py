"""Create SimpleStream-style OVO chunk videos from source videos."""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import cv2
from tqdm import tqdm


FORWARD_TASKS = {"REC", "SSR", "CRR"}


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


def chunk_video(src_path: Path, dst_path: Path, end_time: float, overwrite: bool) -> bool:
    if dst_path.exists() and dst_path.stat().st_size > 0 and not overwrite:
        existing = cv2.VideoCapture(str(dst_path))
        ok = existing.isOpened() and int(existing.get(cv2.CAP_PROP_FRAME_COUNT) or 0) > 0
        existing.release()
        if ok:
            return False
        print(f"[chunk] removing corrupt existing chunk: {dst_path.name}", flush=True)
        dst_path.unlink()
    if dst_path.exists() and dst_path.stat().st_size == 0:
        dst_path.unlink()

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

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[chunk] start {dst_path.name}: {src_path.name}, "
        f"target_frames={end_frame}, end_time={end_time:.2f}s",
        flush=True,
    )
    writer = cv2.VideoWriter(
        str(dst_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create chunk video: {dst_path}")

    written = 0
    last_report = time.monotonic()
    try:
        while written < end_frame:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            written += 1
            now = time.monotonic()
            if now - last_report >= 10.0:
                pct = 100.0 * written / end_frame if end_frame else 0.0
                print(
                    f"[chunk] {dst_path.name}: {written}/{end_frame} frames ({pct:.1f}%)",
                    flush=True,
                )
                last_report = now
    finally:
        writer.release()
        cap.release()

    if written == 0:
        raise RuntimeError(f"No frames written for {dst_path} from {src_path}")
    print(f"[chunk] done {dst_path.name}: {written} frames", flush=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk an OVO-Bench annotation subset.")
    parser.add_argument("--anno_path", required=True)
    parser.add_argument("--src_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow_missing", action="store_true")
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
        did_create = chunk_video(src_path, dst_path, end_time, args.overwrite)
        created += int(did_create)
        skipped += int(not did_create)

    print(f"Created {created} chunks, skipped {skipped} existing chunks in {output_dir}")


if __name__ == "__main__":
    main()
