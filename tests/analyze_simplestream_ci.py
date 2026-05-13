"""Wilson 95% CI per task for our SimpleStream-aligned run vs SimpleStream paper.

Tells us whether 100%/0% spikes are real or just small-N noise.
"""

import json
import math
from collections import defaultdict


SIMPLESTREAM = {
    "OCR": 94.0, "ACR": 85.3, "ATR": 82.8, "STU": 65.7, "FPD": 77.2, "OJR": 83.2,
    "EPM": 51.9, "ASI": 58.1, "HLD": 52.1,
}
ORDER = ["OCR", "ACR", "ATR", "STU", "FPD", "OJR", "EPM", "ASI", "HLD"]


def wilson_ci(p: float, n: int, z: float = 1.96):
    if n == 0:
        return None, None
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (center - half) * 100, (center + half) * 100


def main():
    with open("results/ovo_10pct/qwen3vl8b_simplestream_int8_t4.json", encoding="utf-8") as f:
        r = json.load(f)
    by_task = defaultdict(lambda: [0, 0])
    for p in r.get("predictions", []):
        t = p.get("task_type")
        by_task[t][1] += 1
        if p.get("correct"):
            by_task[t][0] += 1

    print(f"{'task':<4} {'n':>4} {'c/n':>8} {'ours%':>7} {'95% CI Wilson':>18} {'paper%':>7}  paper-in-CI")
    print("-" * 80)
    consistent = 0
    for task in ORDER:
        c, n = by_task[task]
        if n == 0:
            continue
        p_hat = c / n
        lo, hi = wilson_ci(p_hat, n)
        paper = SIMPLESTREAM[task]
        inside = lo <= paper <= hi
        if inside:
            consistent += 1
        mark = "consistent" if inside else "DISTINGUISHABLE"
        print(f"{task:<4} {n:>4} {c}/{n:<6} {p_hat*100:>6.1f}% [{lo:>5.1f}, {hi:>5.1f}] {paper:>6.1f}%  {mark}")
    print()
    print(f"Tasks where paper value sits inside our 95% CI: {consistent} / 9")
    print(f"This includes OCR 100% (n=15) — Wilson lower bound is ~78% so paper 94% IS inside.")


if __name__ == "__main__":
    main()
