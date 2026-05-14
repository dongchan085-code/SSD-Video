"""Final audit: which paper-reproduction knobs really moved the needle?

Uses the existing 25%-non-HLD + HLD@100% combined predictions and recomputes
under OVO-Bench's official substring scoring (the function paper uses).

Prints both macro averages and side-by-side per-task tables.
"""

import json
from collections import defaultdict
from pathlib import Path


PATHS = [
    ("25%+HLD@100% (substring)",
     "results/ovo_simplestream_fullset/qwen3vl8b_nf4_fast_25pct_plus_hld_full_t4.json"),
]

PAPER = {
    "OCR": 94.0, "ACR": 85.3, "ATR": 82.8, "STU": 65.7, "FPD": 77.2, "OJR": 83.2,
    "EPM": 51.9, "ASI": 58.1, "HLD": 52.1,
}
RT = ["OCR", "ACR", "ATR", "STU", "FPD", "OJR"]
BW = ["EPM", "ASI", "HLD"]


def substring_correct(response, gt_int):
    if response is None or not str(response).strip():
        return False
    letter = chr(65 + int(gt_int))
    return letter in str(response)


def main():
    with open(PATHS[0][1], encoding="utf-8") as f:
        r = json.load(f)
    preds = r.get("predictions", [])

    by_task_regex = defaultdict(lambda: [0, 0])
    by_task_substr = defaultdict(lambda: [0, 0])
    for p in preds:
        t = p.get("task_type")
        if t not in PAPER:
            continue
        by_task_regex[t][0] += 1 if p.get("correct") else 0
        by_task_regex[t][1] += 1
        by_task_substr[t][0] += 1 if substring_correct(p.get("answer_text"), p["ground_truth"]) else 0
        by_task_substr[t][1] += 1

    def macro(d, keys):
        vals = [100.0 * d[k][0] / d[k][1] for k in keys if d[k][1] > 0]
        return sum(vals) / len(vals)

    print("=" * 76)
    print("Per-task comparison (25% non-HLD + HLD@100% predictions)")
    print("=" * 76)
    print(f"{'task':<5}{'n':>5}{'paper':>9}{'regex%':>10}{'subst%':>10}{'Δ paper-subst':>16}")
    print("-" * 76)
    for task in RT + BW:
        n = by_task_regex[task][1]
        if n == 0:
            continue
        paper = PAPER[task]
        r_pct = 100.0 * by_task_regex[task][0] / n
        s_pct = 100.0 * by_task_substr[task][0] / n
        delta = s_pct - paper
        print(f"{task:<5}{n:>5}{paper:>8.1f}%{r_pct:>9.2f}%{s_pct:>9.2f}%{delta:>+12.2f}pp")

    print()
    rt_r = macro(by_task_regex, RT)
    rt_s = macro(by_task_substr, RT)
    bw_r = macro(by_task_regex, BW)
    bw_s = macro(by_task_substr, BW)
    print(f"Realtime macro: regex {rt_r:.2f}%, substr {rt_s:.2f}%   (paper 81.4%, Δ {rt_s - 81.4:+.2f}pp)")
    print(f"Backward macro: regex {bw_r:.2f}%, substr {bw_s:.2f}%   (paper 54.0%, Δ {bw_s - 54.0:+.2f}pp)")
    print(f"(RT+BW)/2:      regex {(rt_r + bw_r) / 2:.2f}%, substr {(rt_s + bw_s) / 2:.2f}%   "
          f"(paper 67.7%, Δ {(rt_s + bw_s) / 2 - 67.7:+.2f}pp)")


if __name__ == "__main__":
    main()
