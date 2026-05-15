"""Compare two OVO-Bench result JSONs: overall + per-task accuracy table.

Usage:
    python tests/compare_1pct.py [FILE_A [FILE_B [LABEL_A [LABEL_B]]]]

Defaults to the committed 1%-subset baseline pair when no args are given.
"""

import argparse
import json
import sys
from pathlib import Path


_DEFAULT_A = "results/ovo_1pct/ovo_base_qwen2vl2b_gpu_1pct.json"
_DEFAULT_B = "results/ovo_1pct/qwen3vl8b_nf4_t4.json"

_SUMMARY_KEYS = [
    "overall_accuracy",
    "lock_accuracy",
    "fork_accuracy",
    "realtime_accuracy",
    "backward_accuracy",
    "forward_accuracy",
    "rt_bwd_avg",
    "ovo_total_avg_3way",
    "mean_latency_ms",
    "peak_gpu_memory_gb",
    "throughput_samples_per_sec",
]


def _load(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        sys.exit(f"File not found: {p}")
    with open(p) as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("file_a", nargs="?", default=_DEFAULT_A, help="First result JSON")
    parser.add_argument("file_b", nargs="?", default=_DEFAULT_B, help="Second result JSON")
    parser.add_argument("label_a", nargs="?", default=None, help="Label for file_a (default: filename stem)")
    parser.add_argument("label_b", nargs="?", default=None, help="Label for file_b (default: filename stem)")
    args = parser.parse_args()

    label_a = args.label_a or Path(args.file_a).stem
    label_b = args.label_b or Path(args.file_b).stem

    prev = _load(args.file_a)
    cur = _load(args.file_b)

    col = max(len(label_a), len(label_b), 14)
    fmt = f"{{:<25}} {{:<{col}}} {{:<{col}}}"
    print(fmt.format("metric", label_a, label_b))
    print("-" * (25 + col * 2 + 2))
    for k in _SUMMARY_KEYS:
        a, b = prev.get(k), cur.get(k)
        sa = f"{a:.4f}" if isinstance(a, float) else str(a) if a is not None else "-"
        sb = f"{b:.4f}" if isinstance(b, float) else str(b) if b is not None else "-"
        print(fmt.format(k, sa, sb))

    print()
    print("per-task accuracy:")
    print(fmt.format("task", label_a, label_b))
    keys = sorted(
        set(list(prev.get("per_task_accuracy", {}).keys()) + list(cur.get("per_task_accuracy", {}).keys()))
    )
    for t in keys:
        pa = prev.get("per_task_accuracy", {}).get(t)
        pb = cur.get("per_task_accuracy", {}).get(t)
        sa = f"{pa:.3f}" if pa is not None else "-"
        sb = f"{pb:.3f}" if pb is not None else "-"
        print(fmt.format(t, sa, sb))


if __name__ == "__main__":
    main()
