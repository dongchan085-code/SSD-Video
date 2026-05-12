"""Extract only required OVO source videos from split tar parts."""

import argparse
import shutil
import tarfile
import time
from pathlib import Path, PurePosixPath
from typing import List, Optional, Set


class SplitPartReader:
    """Sequential read-only file object over sorted tar part files."""

    def __init__(self, parts: List[Path], report_interval_bytes: int):
        self.parts = parts
        self.index = 0
        self.current = None
        self.total_bytes = sum(part.stat().st_size for part in parts)
        self.bytes_read = 0
        self.report_interval_bytes = report_interval_bytes
        self.last_report_bytes = 0
        self.start_time = time.monotonic()

    def readable(self) -> bool:
        return True

    def _open_next(self) -> bool:
        if self.current is not None:
            self.current.close()
            self.current = None
        if self.index >= len(self.parts):
            return False
        part = self.parts[self.index]
        print(f"[extract] opening {part.name} ({part.stat().st_size / (1024 ** 3):.2f}GB)", flush=True)
        self.current = part.open("rb")
        self.index += 1
        return True

    def _report(self) -> None:
        if self.bytes_read - self.last_report_bytes < self.report_interval_bytes:
            return
        self.last_report_bytes = self.bytes_read
        elapsed = max(time.monotonic() - self.start_time, 1e-6)
        rate = self.bytes_read / elapsed / (1024 ** 2)
        pct = 100.0 * self.bytes_read / self.total_bytes if self.total_bytes else 0.0
        print(
            f"[extract] scanned {self.bytes_read / (1024 ** 3):.2f}/"
            f"{self.total_bytes / (1024 ** 3):.2f}GB ({pct:.1f}%), {rate:.1f}MB/s",
            flush=True,
        )

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
                self._report()
                if size >= 0:
                    remaining -= len(chunk)
                if size >= 0 and remaining <= 0:
                    break
            else:
                if not self._open_next():
                    break
        return b"".join(chunks)

    def close(self) -> None:
        if self.current is not None:
            self.current.close()
            self.current = None


def load_required(path: Path) -> Set[str]:
    values = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip().replace("\\", "/")
        if stripped:
            values.add(stripped)
    return values


def match_required(member_name: str, required: Set[str]) -> Optional[str]:
    normalized = member_name.replace("\\", "/")
    basename = PurePosixPath(normalized).name
    for item in required:
        if normalized == item or normalized.endswith(f"/{item}"):
            return item
        if "/" not in item and basename == item:
            return item
    return None


def safe_output_path(output_dir: Path, required_name: str) -> Path:
    relative = PurePosixPath(required_name.replace("\\", "/"))
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"Invalid required source path: {required_name}")
    return output_dir.joinpath(*relative.parts)


def extract_subset(
    parts: List[Path],
    required: Set[str],
    output_dir: Path,
    progress_interval_gb: float,
) -> Set[str]:
    found = set()
    output_dir.mkdir(parents=True, exist_ok=True)
    reader = SplitPartReader(
        parts,
        report_interval_bytes=max(1, int(progress_interval_gb * (1024 ** 3))),
    )
    try:
        with tarfile.open(fileobj=reader, mode="r|*") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                matched_name = match_required(member.name, required)
                if matched_name is None:
                    continue

                dst = safe_output_path(output_dir, matched_name)
                dst.parent.mkdir(parents=True, exist_ok=True)
                src = tar.extractfile(member)
                if src is None:
                    continue
                with src, dst.open("wb") as out:
                    shutil.copyfileobj(src, out, length=1024 * 1024)
                found.add(matched_name)
                print(f"[extract] extracted {len(found)}/{len(required)}: {matched_name}", flush=True)
    finally:
        reader.close()
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract required OVO source videos from split tar parts.")
    parser.add_argument("--parts_dir", required=True)
    parser.add_argument("--required_sources", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--allow_missing", action="store_true")
    parser.add_argument("--progress_interval_gb", type=float, default=1.0)
    args = parser.parse_args()

    parts_dir = Path(args.parts_dir)
    parts = sorted(parts_dir.glob("src_videos.tar.part*"))
    if not parts:
        raise SystemExit(f"No src_videos.tar.part* files found in {parts_dir}")

    required = load_required(Path(args.required_sources))
    output_dir = Path(args.output_dir)
    found = extract_subset(parts, required, output_dir, args.progress_interval_gb)
    missing = sorted(required - found)
    if missing and not args.allow_missing:
        raise SystemExit(f"Missing {len(missing)} required source videos, first few: {missing[:10]}")

    print(f"Extracted {len(found)} source videos to {output_dir}")


if __name__ == "__main__":
    main()
