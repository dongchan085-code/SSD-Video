"""Download the OVO-Bench source-video parts and annotation to D: by default."""

import argparse
import json
import shutil
import urllib.request
from pathlib import Path


HF_REPO_ID = "JoeLeelyf/OVO-Bench"
ANNOTATION_URL = "https://raw.githubusercontent.com/joeleelyf/ovo-bench/main/data/ovo_bench_new.json"
SRC_PART_PATTERN = "src_videos.tar.part*"
EXPECTED_SRC_PARTS_GB = 43.16


def gb(num_bytes: int) -> float:
    return num_bytes / (1024 ** 3)


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def download_annotation(annotation_path: Path) -> None:
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    if annotation_path.exists() and annotation_path.stat().st_size > 0:
        return
    with urllib.request.urlopen(ANNOTATION_URL, timeout=60) as response:
        data = response.read()
    parsed = json.loads(data.decode("utf-8"))
    annotation_path.write_text(json.dumps(parsed, indent=2, ensure_ascii=False), encoding="utf-8")


def download_src_parts(parts_dir: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("Install huggingface_hub before downloading OVO-Bench parts.") from exc

    parts_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        allow_patterns=[SRC_PART_PATTERN],
        local_dir=str(parts_dir),
        resume_download=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download OVO-Bench source-video parts.")
    parser.add_argument("--data_root", default="D:/ssd_video_data")
    parser.add_argument("--parts_dir", default=None)
    parser.add_argument("--anno_path", default=None)
    parser.add_argument("--max_gb", type=float, default=100.0)
    parser.add_argument("--skip_parts", action="store_true", help="Only download the annotation file.")
    parser.add_argument("--dry_run", action="store_true", help="Print the size plan without downloading.")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    parts_dir = Path(args.parts_dir) if args.parts_dir else data_root / "ovo_src_parts"
    anno_path = Path(args.anno_path) if args.anno_path else data_root / "ovo_bench_new.json"

    required_gb = EXPECTED_SRC_PARTS_GB
    if required_gb > args.max_gb:
        raise SystemExit(
            f"Source parts require about {required_gb:.2f}GB, above --max_gb={args.max_gb:.2f}GB."
        )

    free = shutil.disk_usage(str(data_root.anchor or ".")).free if data_root.anchor else shutil.disk_usage(".").free
    part_plan = (
        f"annotation only; source parts are skipped"
        if args.skip_parts
        else f"annotation plus {SRC_PART_PATTERN} from {HF_REPO_ID}"
    )
    print(
        f"Plan: download {part_plan}. "
        f"Expected source-video parts if enabled: {required_gb:.2f}GB. "
        f"Free on target drive: {gb(free):.2f}GB."
    )
    if args.dry_run:
        return

    data_root.mkdir(parents=True, exist_ok=True)
    download_annotation(anno_path)
    if not args.skip_parts:
        download_src_parts(parts_dir)

    actual_parts_gb = gb(directory_size(parts_dir))
    print(f"Annotation: {anno_path}")
    print(f"Source parts: {parts_dir} ({actual_parts_gb:.2f}GB currently present)")


if __name__ == "__main__":
    main()
