"""Combine 25%-sample run with the HLD-100% run into a single results JSON.

Inputs:
  - results/ovo_simplestream_fullset/qwen3vl8b_nf4_fast_25pct_t4.snapshot.json
    (12 tasks at 25% sample rate)
  - results/ovo_simplestream_fullset/qwen3vl8b_nf4_fast_hld_full_t4.json
    (HLD only, 100% / all 186 source HLD samples)

Output:
  - results/ovo_simplestream_fullset/qwen3vl8b_nf4_fast_25pct_plus_hld_full_t4.json
    11 tasks @ 25% + HLD @ 100%, with recomputed per-task / Lock / Fork /
    overall metrics.
"""

import json
from collections import defaultdict
from pathlib import Path

from ssd_vlm.eval_metrics import summarize_ovo_predictions
from ssd_vlm.data.ovo_bench_dataset import FORK_TASKS, LOCK_TASKS


SRC_25 = Path("results/ovo_simplestream_fullset/qwen3vl8b_nf4_fast_25pct_t4.snapshot.json")
SRC_HLD = Path("results/ovo_simplestream_fullset/qwen3vl8b_nf4_fast_hld_full_t4.json")
OUT = Path("results/ovo_simplestream_fullset/qwen3vl8b_nf4_fast_25pct_plus_hld_full_t4.json")


def main():
    with SRC_25.open(encoding="utf-8") as f:
        r25 = json.load(f)
    with SRC_HLD.open(encoding="utf-8") as f:
        rhld = json.load(f)

    non_hld_25 = [p for p in r25.get("predictions", []) if p.get("task_type") != "HLD"]
    hld_100 = list(rhld.get("predictions", []))

    combined_preds = non_hld_25 + hld_100

    # Verify HLD count
    hld_n = sum(1 for p in combined_preds if p.get("task_type") == "HLD")
    print(f"non-HLD (25% sample): {len(non_hld_25)}")
    print(f"HLD (100% / fullset): {len(hld_100)} (expected 186)")
    print(f"combined predictions: {len(combined_preds)}")

    decoding_meta = r25.get("decoding_meta") or {}
    streaming_meta = r25.get("streaming_meta") or {}
    peak_gpu_memory_gb = max(
        r25.get("peak_gpu_memory_gb") or 0,
        rhld.get("peak_gpu_memory_gb") or 0,
    ) or None

    results = summarize_ovo_predictions(
        combined_preds,
        lock_tasks=LOCK_TASKS,
        fork_tasks=FORK_TASKS,
        decoding_meta=decoding_meta,
        streaming_meta=streaming_meta,
        save_predictions=True,
        peak_gpu_memory_gb=peak_gpu_memory_gb,
    )

    results["composition"] = {
        "note": "11 tasks @ 25% stratified sample + HLD @ 100% (all 186 HLD samples)",
        "non_hld_predictions_count": len(non_hld_25),
        "hld_predictions_count": len(hld_100),
        "sample_seed": 42,
        "sample_ratio_non_hld": 0.25,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nWrote combined results to {OUT}")

    # Print summary
    print()
    print(f"Overall: {results['overall_accuracy']*100:.2f}%")
    print(f"Lock:    {results['lock_accuracy']*100:.2f}%")
    print(f"Fork:    {results['fork_accuracy']*100:.2f}%")
    print()
    for task, acc in sorted(results["per_task_accuracy"].items()):
        print(f"  {task}: {acc*100:.2f}%")


if __name__ == "__main__":
    main()
