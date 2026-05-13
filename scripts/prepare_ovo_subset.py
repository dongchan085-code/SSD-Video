"""Create a stratified OVO-Bench subset manifest for SimpleStream checks.

The public OVO-Bench Hugging Face repo stores videos as tar parts.  This script
keeps the annotation subset small and writes exact source/chunk manifests so the
follow-up extraction step can pull only the required videos from the source tar.
"""

import argparse
import copy
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List

from ssd_vlm.simplestream import (
    BACKWARD_TASK_SET as BACKWARD_TASKS,
    FORWARD_TASK_SET as FORWARD_TASKS,
    REAL_TIME_TASK_SET as REALTIME_TASKS,
)


def source_name(annotation: Dict[str, Any]) -> str:
    value = annotation.get("video") or annotation.get("video_path")
    if value:
        return str(value).replace("\\", "/")
    return f"{annotation.get('id')}.mp4"


def chunk_names(annotation: Dict[str, Any]) -> List[str]:
    video_id = str(annotation["id"])
    task = annotation.get("task", annotation.get("task_type", ""))
    if task in FORWARD_TASKS:
        test_info = annotation.get("test_info") or []
        return [f"{video_id}_{idx}.mp4" for idx, _ in enumerate(test_info)]
    return [f"{video_id}.mp4"]


def query_units(annotation: Dict[str, Any]) -> int:
    task = annotation.get("task", annotation.get("task_type", ""))
    if task in FORWARD_TASKS:
        return max(1, len(annotation.get("test_info") or []))
    return 1


def task_group(task: str) -> str:
    if task in BACKWARD_TASKS:
        return "backward"
    if task in REALTIME_TASKS:
        return "realtime"
    if task in FORWARD_TASKS:
        return "forward"
    return "unknown"


def attach_chunk_relpaths(annotation: Dict[str, Any]) -> Dict[str, Any]:
    enriched = copy.deepcopy(annotation)
    video_id = str(enriched["id"])
    task = enriched.get("task", enriched.get("task_type", ""))
    if task in FORWARD_TASKS:
        test_info = enriched.get("test_info") or []
        for idx, item in enumerate(test_info):
            item["video_relpath"] = f"chunked_videos/{video_id}_{idx}.mp4"
    else:
        enriched["video_relpath"] = f"chunked_videos/{video_id}.mp4"
    return enriched


def stable_sample(items: List[Dict[str, Any]], n: int, rng: random.Random) -> List[Dict[str, Any]]:
    if n >= len(items):
        return list(items)
    sampled = rng.sample(items, n)
    return sorted(sampled, key=lambda row: str(row.get("id", "")))


def select_subset(
    annotations: List[Dict[str, Any]],
    ratio: float,
    seed: int,
    min_per_task: int,
    max_annotations: int = 0,
) -> List[Dict[str, Any]]:
    if not 0 < ratio <= 1:
        raise ValueError("--ratio must be in (0, 1].")

    rng = random.Random(seed)
    by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in annotations:
        if item.get("id") is None:
            continue
        by_task[str(item.get("task", item.get("task_type", "unknown")))].append(item)

    selected: List[Dict[str, Any]] = []
    for task, items in sorted(by_task.items()):
        items = sorted(items, key=lambda row: str(row.get("id", "")))
        target = max(min_per_task, math.ceil(len(items) * ratio))
        target = min(target, len(items))
        selected.extend(stable_sample(items, target, rng))

    if max_annotations and len(selected) > max_annotations:
        selected = stable_sample(selected, max_annotations, rng)

    return sorted(selected, key=lambda row: (str(row.get("task", "")), str(row.get("id", ""))))


def write_lines(path: Path, values: Iterable[str]) -> None:
    path.write_text("\n".join(sorted(set(values))) + "\n", encoding="utf-8")


def build_report(annotations: List[Dict[str, Any]]) -> Dict[str, Any]:
    task_counts = Counter(str(item.get("task", item.get("task_type", "unknown"))) for item in annotations)
    group_counts = Counter(task_group(task) for task in task_counts for _ in range(task_counts[task]))
    query_counts = Counter()
    for item in annotations:
        query_counts[task_group(str(item.get("task", item.get("task_type", ""))))] += query_units(item)
    return {
        "annotations": len(annotations),
        "query_units": sum(query_units(item) for item in annotations),
        "source_videos": len({source_name(item) for item in annotations}),
        "chunk_videos": sum(len(chunk_names(item)) for item in annotations),
        "tasks": dict(sorted(task_counts.items())),
        "groups_by_annotation": dict(sorted(group_counts.items())),
        "groups_by_query_unit": dict(sorted(query_counts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an OVO-Bench subset manifest.")
    parser.add_argument("--anno_path", required=True, help="Path to ovo_bench_new.json")
    parser.add_argument("--output_dir", required=True, help="Subset output directory")
    parser.add_argument("--ratio", type=float, required=True, help="Fraction per task, e.g. 0.01 or 0.10")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_per_task", type=int, default=1)
    parser.add_argument("--max_annotations", type=int, default=0)
    args = parser.parse_args()

    anno_path = Path(args.anno_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    annotations = json.loads(anno_path.read_text(encoding="utf-8"))
    if not isinstance(annotations, list):
        raise ValueError(f"Expected a list annotation file: {anno_path}")

    selected = select_subset(
        annotations=annotations,
        ratio=args.ratio,
        seed=args.seed,
        min_per_task=args.min_per_task,
        max_annotations=args.max_annotations,
    )
    enriched = [attach_chunk_relpaths(item) for item in selected]

    subset_path = output_dir / "ovo_bench_subset.json"
    subset_path.write_text(json.dumps(enriched, indent=2, ensure_ascii=False), encoding="utf-8")

    write_lines(output_dir / "required_sources.txt", (source_name(item) for item in enriched))
    write_lines(output_dir / "required_chunks.txt", (name for item in enriched for name in chunk_names(item)))

    report = build_report(enriched)
    report.update({
        "ratio": args.ratio,
        "seed": args.seed,
        "annotation_path": str(anno_path),
        "subset_path": str(subset_path),
    })
    (output_dir / "subset_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
