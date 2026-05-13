"""Compare OVO benchmark metrics across 1% (Qwen2VL-2B baseline / Qwen3VL-8B NF4)
and 10% (Qwen3VL-8B NF4) subsets."""

import json
from collections import defaultdict


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def per_task(r):
    preds = r.get("predictions", [])
    counts = defaultdict(lambda: [0, 0])
    for p in preds:
        t = p.get("task_type")
        counts[t][1] += 1
        if p.get("correct"):
            counts[t][0] += 1
    return {t: (c, n) for t, (c, n) in counts.items()}


def main():
    runs = [
        ("Qwen2VL-2B (1%)", "results/ovo_1pct/ovo_base_qwen2vl2b_gpu_1pct.json"),
        ("Qwen3VL-8B NF4 (1%)", "results/ovo_1pct/qwen3vl8b_nf4_t4.json"),
        ("Qwen3VL-8B NF4 (10%)", "results/ovo_10pct/qwen3vl8b_nf4_t4.json"),
    ]
    data = [(label, load(path)) for label, path in runs]

    print("=" * 90)
    print("Aggregate metrics:")
    header = ["metric"] + [label for label, _ in data]
    widths = [28, 22, 22, 22]

    def row(values):
        print("".join(str(v).ljust(w) for v, w in zip(values, widths)))

    row(header)
    print("-" * sum(widths))
    keys = [
        ("overall_accuracy", "overall (raw)"),
        ("num_correct", "  correct / total"),
        ("lock_accuracy", "lock (realtime)"),
        ("fork_accuracy", "fork (backward)"),
        ("forward_accuracy", "forward"),
        ("rt_bwd_avg", "lock+fork avg"),
        ("ovo_total_avg_3way", "3-way avg"),
        ("peak_gpu_memory_gb", "peak GPU (GB)"),
        ("mean_latency_ms", "mean latency (ms)"),
        ("throughput_samples_per_sec", "throughput (samp/s)"),
    ]
    for k, label in keys:
        vals = [label]
        for _, r in data:
            v = r.get(k)
            if k == "num_correct":
                v = f"{r.get('num_correct')}/{r.get('num_total')}"
            elif isinstance(v, float):
                v = f"{v:.4f}"
            vals.append(str(v))
        row(vals)

    print()
    print("=" * 90)
    print("Per-task accuracy (count correct / total):")
    all_tasks = sorted(set().union(*[per_task(r).keys() for _, r in data]))
    row(["task"] + [label for label, _ in data])
    print("-" * sum(widths))
    for t in all_tasks:
        row_data = [t]
        for _, r in data:
            pt = per_task(r)
            if t in pt:
                c, n = pt[t]
                row_data.append(f"{c}/{n} ({c/n:.3f})")
            else:
                row_data.append("-")
        row(row_data)


if __name__ == "__main__":
    main()
