"""Stream HF OVO chunked video tar parts and write PNG replay caches.

This is the low-disk full OVO bootstrap path for the T4 VM. It downloads one
tar part at a time, extracts each mp4 to a temporary file, writes the
SimpleStream/Qwen3 recent-frame PNG cache, then deletes the temporary mp4
before moving to the next member.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import threading
import time
from pathlib import Path
from queue import Queue
from typing import Dict, Iterable, List, Optional, Set

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.download_extract_chunked import (  # noqa: E402
    RollingPartReader,
    background_downloader,
    iter_downloaded_parts,
    list_remote_parts,
    load_include_names,
    member_include_key,
)
from scripts.precompute_ovo_simplestream_frames import (  # noqa: E402
    _cache_complete,
    write_precomputed_cache,
)
from ssd_vlm.data.ovo_bench_dataset import OVOBenchDataset  # noqa: E402


def _print(message: str) -> None:
    print(message, flush=True)


def _sample_index(dataset: OVOBenchDataset) -> Dict[str, dict]:
    index: Dict[str, dict] = {}
    for sample in dataset.samples:
        index.setdefault(str(sample["video_id"]), sample)
    return index


def _extract_member_to_temp(tar: tarfile.TarFile, member: tarfile.TarInfo, target: Path) -> None:
    src = tar.extractfile(member)
    if src is None:
        raise RuntimeError(f"tar member has no file object: {member.name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    try:
        with src, tmp.open("wb") as out:
            shutil.copyfileobj(src, out, length=4 * 1024 * 1024)
        tmp.replace(target)
    finally:
        if tmp.exists():
            tmp.unlink()


def _selected(include_names: Optional[Set[str]], key: str) -> bool:
    return include_names is None or key in include_names


def stream_precompute(
    *,
    repo_id: str,
    tar_glob: str,
    parts_dir: Path,
    work_dir: Path,
    output_dir: Path,
    dataset: OVOBenchDataset,
    include_names: Optional[Set[str]],
    max_parts_ahead: int,
    recent_frames_only: int,
    chunk_duration: float,
    fps: float,
    overwrite: bool,
) -> dict:
    remote_parts = list_remote_parts(repo_id, tar_glob)
    _print(f"Remote parts to fetch: {len(remote_parts)}")
    for name in remote_parts:
        _print(f"  {name}")

    samples_by_id = _sample_index(dataset)
    parts_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    queue: "Queue[Path]" = Queue(maxsize=max(1, int(max_parts_ahead)))
    error_box: List[Exception] = []
    thread = threading.Thread(
        target=background_downloader,
        args=(repo_id, remote_parts, parts_dir, queue, error_box),
        daemon=True,
    )
    thread.start()

    reader = RollingPartReader(iter_downloaded_parts(queue), total_bytes=int(152 * 1024 ** 3))
    started = time.perf_counter()
    seen = 0
    selected = 0
    cached = 0
    skipped = 0
    missing = 0
    failed = 0

    try:
        with tarfile.open(fileobj=reader, mode="r|*") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                key = member_include_key(member.name)
                if key is None or not key.lower().endswith(".mp4"):
                    continue
                seen += 1
                if not _selected(include_names, key):
                    continue
                video_id = Path(key).stem
                sample = samples_by_id.get(video_id)
                if sample is None:
                    missing += 1
                    continue
                selected += 1
                cache_dir = output_dir / video_id
                if not overwrite and _cache_complete(cache_dir, recent_frames_only, chunk_duration, fps):
                    skipped += 1
                    if selected == 1 or selected % 25 == 0:
                        _print(f"PROGRESS selected={selected} cached={cached} skipped={skipped} failed={failed}")
                    continue

                temp_video = work_dir / key
                try:
                    _extract_member_to_temp(tar, member, temp_video)
                    frames = write_precomputed_cache(
                        sample=sample,
                        video_path=temp_video,
                        cache_dir=cache_dir,
                        recent_frames_only=recent_frames_only,
                        chunk_duration=chunk_duration,
                        fps=fps,
                        overwrite=overwrite,
                    )
                    cached += 1
                    status = "CACHE"
                except Exception as exc:
                    failed += 1
                    frames = 0
                    status = "FAIL"
                    _print(f"ERROR key={key} id={video_id} error={exc!r}")
                finally:
                    if temp_video.exists():
                        temp_video.unlink()

                if selected == 1 or selected % 25 == 0 or status == "FAIL":
                    elapsed = time.perf_counter() - started
                    _print(
                        "PROGRESS "
                        f"seen={seen} selected={selected} status={status} id={video_id} "
                        f"frames={frames} cached={cached} skipped={skipped} missing={missing} "
                        f"failed={failed} elapsed_s={elapsed:.1f}"
                    )
    finally:
        reader.close()

    if error_box:
        raise error_box[0]
    thread.join(timeout=30)
    if thread.is_alive():
        _print("[warn] downloader thread still alive after 30s; abandoning")
    if error_box:
        raise error_box[0]

    summary = {
        "seen": seen,
        "selected": selected,
        "cached": cached,
        "skipped": skipped,
        "missing": missing,
        "failed": failed,
        "output_dir": str(output_dir),
    }
    _print("STREAM_PRECOMPUTE_DONE " + json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-id", default="JoeLeelyf/OVO-Bench")
    parser.add_argument("--tar-glob", default="chunked_videos.tar.part*")
    parser.add_argument("--parts-dir", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-path", default="D:/ssd_video_data")
    parser.add_argument("--anno-path", default="D:/ssd_video_data/ovo_bench_new.json")
    parser.add_argument("--chunked-dir", default="D:/ssd_video_data/chunked_videos")
    parser.add_argument("--include-list")
    parser.add_argument("--max-parts-ahead", type=int, default=1)
    parser.add_argument("--recent-frames-only", type=int, default=4)
    parser.add_argument("--chunk-duration", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

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
    include_names = load_include_names(args.include_list)
    stream_precompute(
        repo_id=args.repo_id,
        tar_glob=args.tar_glob,
        parts_dir=Path(args.parts_dir),
        work_dir=Path(args.work_dir),
        output_dir=Path(args.output_dir),
        dataset=dataset,
        include_names=include_names,
        max_parts_ahead=args.max_parts_ahead,
        recent_frames_only=args.recent_frames_only,
        chunk_duration=args.chunk_duration,
        fps=args.fps,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
