"""Compare our T4 int8 SimpleStream-aligned 10% run vs the published numbers.

SimpleStream Qwen3-VL-8B with recent_frames_only=4 (4f) from the project
page + Table 1 in arXiv:2604.02317v1:

| task | SimpleStream |
|------|--------------|
| OCR  | 94.0         |
| ACR  | 85.3         |
| ATR  | 82.8         |
| STU  | 65.7         |
| FPD  | 77.2         |
| OJR  | 83.2         |
| Realtime Avg | 81.4 |
| EPM  | 51.9         |
| ASI  | 58.1         |
| HLD  | 52.1         |
| Backward Avg | 54.0 |
| Overall (RT + Bwd / 2 reported)     | 67.7 |

REC/SSR/CRR not reported separately in the main table. We omit those rows
when comparing.
"""

import json
from collections import defaultdict


REAL_TIME = ["OCR", "ACR", "ATR", "STU", "FPD", "OJR"]
BACKWARD = ["EPM", "ASI", "HLD"]

SIMPLESTREAM = {
    "OCR": 94.0, "ACR": 85.3, "ATR": 82.8, "STU": 65.7, "FPD": 77.2, "OJR": 83.2,
    "EPM": 51.9, "ASI": 58.1, "HLD": 52.1,
}


def per_task(r):
    by = defaultdict(lambda: [0, 0])
    for p in r.get("predictions", []):
        t = p.get("task_type")
        by[t][1] += 1
        if p.get("correct"):
            by[t][0] += 1
    return {t: 100.0 * c / n for t, (c, n) in by.items() if n}


def macro(values, keys):
    accs = [values[k] for k in keys if k in values]
    return sum(accs) / len(accs) if accs else None


def main():
    runs = [
        ("25% sample (snapshot)", "results/ovo_simplestream_fullset/qwen3vl8b_nf4_fast_25pct_t4.snapshot.json"),
        ("25% + HLD@100%",        "results/ovo_simplestream_fullset/qwen3vl8b_nf4_fast_25pct_plus_hld_full_t4.json"),
    ]
    data = []
    for label, path in runs:
        with open(path, encoding="utf-8") as f:
            data.append((label, per_task(json.load(f))))

    print(f"{'task':<5} {'paper':>8} " + " ".join(f"{label[:24]:>26}" for label, _ in data) + " (Δ vs paper)")
    print("-" * (5 + 9 + 28 * len(data) + 14))

    for task in REAL_TIME + BACKWARD:
        paper = SIMPLESTREAM[task]
        cells = [f"{task:<5} {paper:>7.1f}%"]
        for _, ours in data:
            v = ours.get(task)
            if v is None:
                cells.append(f"{'-':>26}")
                continue
            cells.append(f"{v:>7.1f}% (Δ {v - paper:+6.1f}pp)        ")
        print(" ".join(cells))

    print()
    for label, ours in data:
        rt = macro(ours, REAL_TIME)
        bw = macro(ours, BACKWARD)
        tot = ((rt or 0) + (bw or 0)) / 2 if rt and bw else None
        paper_rt, paper_bw, paper_tot = 81.4, 54.0, 67.7
        print(f"[{label}]")
        if rt is not None:
            print(f"  Realtime macro: {rt:5.1f}%  (paper {paper_rt:.1f}, Δ {rt - paper_rt:+5.1f}pp)")
        if bw is not None:
            print(f"  Backward macro: {bw:5.1f}%  (paper {paper_bw:.1f}, Δ {bw - paper_bw:+5.1f}pp)")
        if tot is not None:
            print(f"  RT+BW   /2    : {tot:5.1f}%  (paper {paper_tot:.1f}, Δ {tot - paper_tot:+5.1f}pp)")
        print()


if __name__ == "__main__":
    main()
