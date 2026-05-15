"""Stream-download HF OVO-Bench chunked_videos.tar.part* and extract on the fly.

Why this exists
---------------
The OVO-Bench HF dataset publishes a pre-chunked archive
(``chunked_videos.tar.part{aa..ao}`` ~ 152 GB total). On a 176 GB D:\\ disk we
cannot keep both the raw tar parts and the extracted ``chunked_videos/``
tree at the same time. This script downloads one part at a time, feeds it
into a streaming ``tarfile`` reader, and deletes each part as soon as the
extractor finishes reading it — peak disk usage stays near
``extracted_so_far + 1-2 tar parts`` rather than ``152 GB + 152 GB``.

Usage
-----
    python -u scripts/download_extract_chunked.py \\
        --repo_id JoeLeelyf/OVO-Bench \\
        --parts_dir D:/ssd_video_data/_chunked_parts \\
        --output_dir D:/ssd_video_data/chunked_videos \\
        --tar_glob "chunked_videos.tar.part*"

Notes
-----
- Requires ``huggingface_hub``. Set ``HF_HOME=D:/hf_cache`` if you want the
  per-part hub cache symlinks to land on D:\\.
- The extractor verifies that every produced file lives under
  ``chunked_videos/`` inside the tar (the HF archive root). Members with
  any ``..`` component are rejected.
- If the script crashes mid-way, restart it — already-downloaded tar parts
  that survived the crash are skipped, and already-extracted output files
  are kept (we never overwrite).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile
import threading
import time
from pathlib import Path, PurePosixPath
from queue import Queue
from typing import Iterable, List, Optional


def gb(num_bytes: int) -> float:
    return num_bytes / (1024 ** 3)


def list_remote_parts(repo_id: str, tar_glob: str) -> List[str]:
    from huggingface_hub import HfApi

    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    import fnmatch

    matches = sorted(f for f in files if fnmatch.fnmatch(f, tar_glob))
    if not matches:
        raise SystemExit(f"No remote files matched {tar_glob!r} in {repo_id!r}")
    return matches


def download_part(repo_id: str, filename: str, dest_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    target = dest_dir / filename
    if target.exists() and target.stat().st_size > 0:
        return target
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download] {filename} ...", flush=True)
    t0 = time.monotonic()
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=str(dest_dir),
    )
    size_mb = target.stat().st_size / (1024 ** 2)
    dt = time.monotonic() - t0
    print(
        f"[download] {filename} done: {size_mb:.0f}MB in {dt:.1f}s ({size_mb / max(dt, 1e-6):.1f} MB/s)",
        flush=True,
    )
    return Path(local_path)


class RollingPartReader:
    """File-like sequential reader over downloaded tar parts.

    Pulls each next path lazily from a Path iterator, reads it fully,
    deletes it, then asks the iterator for the next one. The iterator is
    fed by the downloader thread via a bounded queue, so the producer
    blocks once ``max_parts_ahead`` parts are downloaded — this keeps
    peak disk usage at roughly ``extracted_so_far + max_parts_ahead``.
    """

    def __init__(self, parts_iter: Iterable[Path], total_bytes: Optional[int] = None):
        self.parts_iter = iter(parts_iter)
        self.current: Optional[object] = None  # file handle
        self.current_path: Optional[Path] = None
        self.exhausted = False
        self.total_bytes = total_bytes
        self.bytes_read = 0
        self.last_report_bytes = 0
        self.report_interval = max(1, int((total_bytes or 1) // 50))
        self.start = time.monotonic()

    def readable(self) -> bool:
        return True

    def _open_next(self) -> bool:
        if self.current is not None:
            self.current.close()
            try:
                self.current_path.unlink()
                print(f"[extract] deleted {self.current_path.name} after consumption", flush=True)
            except FileNotFoundError:
                pass
            self.current = None
            self.current_path = None
        if self.exhausted:
            return False
        try:
            part = next(self.parts_iter)
        except StopIteration:
            self.exhausted = True
            return False
        size_mb = part.stat().st_size / (1024 ** 2)
        print(f"[extract] opening {part.name} ({size_mb:.0f}MB)", flush=True)
        self.current = part.open("rb")
        self.current_path = part
        return True

    def _maybe_report(self) -> None:
        if self.bytes_read - self.last_report_bytes < self.report_interval:
            return
        self.last_report_bytes = self.bytes_read
        elapsed = max(time.monotonic() - self.start, 1e-6)
        rate = self.bytes_read / elapsed / (1024 ** 2)
        if self.total_bytes:
            pct = 100.0 * self.bytes_read / self.total_bytes
            print(
                f"[extract] read {gb(self.bytes_read):.2f}/{gb(self.total_bytes):.2f}GB "
                f"({pct:.1f}%, {rate:.1f}MB/s)",
                flush=True,
            )
        else:
            print(f"[extract] read {gb(self.bytes_read):.2f}GB ({rate:.1f}MB/s)", flush=True)

    def read(self, size: int = -1) -> bytes:
        chunks = []
        remaining = size
        while size < 0 or remaining > 0:
            if self.current is None and not self._open_next():
                break
            chunk = self.current.read(remaining if size >= 0 else -1)
            if chunk:
                chunks.append(chunk)
                self.bytes_read += len(chunk)
                self._maybe_report()
                if size >= 0:
                    remaining -= len(chunk)
            else:
                if not self._open_next():
                    break
        return b"".join(chunks)

    def close(self) -> None:
        if self.current is not None:
            self.current.close()
            self.current = None


def safe_output_path(output_dir: Path, member_name: str) -> Optional[Path]:
    normalized = member_name.replace("\\", "/").lstrip("./")
    relative = PurePosixPath(normalized)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        return None
    # Strip the leading "chunked_videos/" if the tar nests under that prefix.
    if relative.parts and relative.parts[0] == "chunked_videos":
        relative = PurePosixPath(*relative.parts[1:])
    if not relative.parts:
        return None
    return output_dir.joinpath(*relative.parts)


def extract_stream(reader: RollingPartReader, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with tarfile.open(fileobj=reader, mode="r|*") as tar:
        for member in tar:
            if not member.isfile():
                continue
            dst = safe_output_path(output_dir, member.name)
            if dst is None:
                print(f"[extract] skipping unsafe member {member.name!r}", flush=True)
                continue
            if dst.exists() and dst.stat().st_size > 0:
                # already extracted in a previous run
                src = tar.extractfile(member)
                if src is not None:
                    src.close()
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(member)
            if src is None:
                continue
            with src, dst.open("wb") as out:
                shutil.copyfileobj(src, out, length=4 * 1024 * 1024)
            count += 1
            if count % 100 == 0:
                print(f"[extract] wrote {count} files (latest: {dst.name})", flush=True)
    return count


def background_downloader(
    repo_id: str,
    remote_parts: List[str],
    parts_dir: Path,
    out_queue: "Queue[Path]",
    error_box: List[Exception],
) -> None:
    try:
        for filename in remote_parts:
            path = download_part(repo_id, filename, parts_dir)
            out_queue.put(path)
        out_queue.put(None)  # sentinel
    except Exception as exc:  # surfaced in the main thread
        error_box.append(exc)
        out_queue.put(None)


def iter_downloaded_parts(out_queue: "Queue[Path]") -> Iterable[Path]:
    while True:
        item = out_queue.get()
        if item is None:
            return
        yield item


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream-download + extract HF OVO chunked_videos tar parts.")
    parser.add_argument("--repo_id", default="JoeLeelyf/OVO-Bench")
    parser.add_argument("--tar_glob", default="chunked_videos.tar.part*")
    parser.add_argument("--parts_dir", required=True, help="Temporary directory for downloaded tar parts (gets emptied)")
    parser.add_argument("--output_dir", required=True, help="Destination directory for extracted chunked videos")
    parser.add_argument(
        "--max_parts_ahead",
        type=int,
        default=2,
        help="How many downloaded tar parts may sit on disk ahead of the extractor",
    )
    args = parser.parse_args()

    parts_dir = Path(args.parts_dir)
    output_dir = Path(args.output_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)

    remote_parts = list_remote_parts(args.repo_id, args.tar_glob)
    print(f"Remote parts to fetch: {len(remote_parts)}", flush=True)
    for name in remote_parts:
        print(f"  {name}", flush=True)

    queue: "Queue[Path]" = Queue(maxsize=max(1, int(args.max_parts_ahead)))
    error_box: List[Exception] = []
    thread = threading.Thread(
        target=background_downloader,
        args=(args.repo_id, remote_parts, parts_dir, queue, error_box),
        daemon=True,
    )
    thread.start()

    # Total bytes ~ 152 GB; not exact, only used for progress reporting.
    reader = RollingPartReader(iter_downloaded_parts(queue), total_bytes=int(152 * 1024 ** 3))
    try:
        count = extract_stream(reader, output_dir)
    finally:
        reader.close()

    # Surface a downloader exception immediately if one was raised — the
    # thread sets error_box before putting its final sentinel, so by the
    # time the reader has exited the box is guaranteed to be populated
    # on the failure path. Don't pay the 300 s join timeout for nothing.
    if error_box:
        raise error_box[0]
    thread.join(timeout=30)
    if thread.is_alive():
        print("[warn] downloader thread still alive after 30s; abandoning", flush=True)
    if error_box:
        raise error_box[0]

    print(f"Extracted {count} files to {output_dir}", flush=True)
    # Best-effort cleanup of the parts_dir scratch (should be empty already).
    try:
        for residual in parts_dir.iterdir():
            print(f"[cleanup] removing leftover {residual}", flush=True)
            if residual.is_file():
                residual.unlink()
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    main()
