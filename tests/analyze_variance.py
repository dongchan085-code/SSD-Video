"""Quantify whether per-task differences vs OVO paper are within sampling noise.

For each task with N samples and observed accuracy p:
- Wilson 95% confidence interval for binomial proportion
- Wald approximation: p +/- 1.96 * sqrt(p*(1-p)/N)
- Z-score for the difference (ours - paper), assuming paper is a population
  parameter (a conservative assumption — paper itself has finite N too).

If the paper value lies inside our 95% CI, the difference is not statistically
distinguishable from noise at this subset size.
"""

import json
import math
from collections import defaultdict


PAPER_QWEN2VL_7B = {
    "EPM": 47.81, "ASI": 35.48, "HLD": 56.08,
    "OCR": 60.40, "ACR": 50.46, "ATR": 56.03, "STU": 47.19, "FPD": 66.34, "OJR": 55.43,
    "REC": 31.66, "SSR": 65.82, "CRR": 48.75,
}

PAPER_TOTAL_N = {
    # From the OVO-Bench paper: total queries per task across the full benchmark.
    "EPM": 297, "ASI": 148, "HLD": 186,
    "OCR": 149, "ACR": 109, "ATR": 116, "STU": 178, "FPD": 101, "OJR": 184,
    "REC": 82, "SSR": 42, "CRR": 48,  # annotation counts; query units multiply this by avg test_info length
}


def wilson_ci(p, n, z=1.96):
    if n == 0:
        return (None, None)
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (center - half, center + half)


def main():
    with open("results/ovo_10pct/qwen3vl8b_nf4_t4.json", encoding="utf-8") as f:
        r = json.load(f)
    by_task = defaultdict(lambda: [0, 0])
    for p in r.get("predictions", []):
        t = p.get("task_type")
        by_task[t][1] += 1
        if p.get("correct"):
            by_task[t][0] += 1

    print(f"{'task':<5} {'n':>4} {'ours%':>7} {'95%CI Wilson':>20} {'paper%':>7} "
          f"{'Δ':>6} {'paper in CI':>12} {'z(noise)':>10}")
    print("-" * 90)

    indistinguishable = 0
    significantly_lower = 0
    significantly_higher = 0
    for task in sorted(by_task):
        c, n = by_task[task]
        p_hat = c / n
        lo, hi = wilson_ci(p_hat, n)
        lo_pct, hi_pct = lo * 100, hi * 100
        paper_pct = PAPER_QWEN2VL_7B.get(task)
        if paper_pct is None:
            continue
        delta = p_hat * 100 - paper_pct
        paper_in_ci = lo_pct <= paper_pct <= hi_pct
        # z-score: assume paper is true population mean; SE ~= sqrt(p*(1-p)/n)
        se = math.sqrt(p_hat * (1 - p_hat) / n) if 0 < p_hat < 1 else float("inf")
        z = delta / 100 / se if se > 0 else float("inf")
        marker = "✓" if paper_in_ci else ("↓" if delta < 0 else "↑")
        print(f"{task:<5} {n:>4} {p_hat*100:>6.1f}% "
              f"[{lo_pct:>5.1f}, {hi_pct:>5.1f}] {paper_pct:>6.1f}% "
              f"{delta:>+5.1f}  {marker + ' ' + str(paper_in_ci):>12}  {z:>+8.2f}")
        if paper_in_ci:
            indistinguishable += 1
        elif delta < 0:
            significantly_lower += 1
        else:
            significantly_higher += 1

    print()
    print(f"Summary:")
    print(f"  paper value INSIDE our 95% CI (noise-indistinguishable): {indistinguishable} / 12 tasks")
    print(f"  significantly LOWER than paper:                          {significantly_lower}")
    print(f"  significantly HIGHER than paper:                         {significantly_higher}")
    print()
    print("Reminder: our 10% subset has tiny per-task N (11-73). For p=50%, n=15 gives")
    print("a 95% CI half-width of ±25.3pp. So a Δ=±10pp is well within sampling noise")
    print("for the perception tasks. Only Δs that exceed the CI are evidence of a real")
    print("setup-driven gap (frames, quantization, model generation).")


if __name__ == "__main__":
    main()
