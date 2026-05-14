"""Inspect raw HLD predictions to find prompt/annotation/output anomalies.

This dumps a handful of HLD prediction records side-by-side with their
ground-truth letter so we can spot:
- Are responses one-letter? (eliminates substring-vs-regex hypothesis)
- Does the model collapse to a single letter (likely 'A' or 'C')? (sign of
  modality dropout / weight regression)
- Is the question/options text identical to SimpleStream's verbatim format?
- Are the ground-truth letters distributed reasonably or skewed?
"""

import json
from collections import Counter
from pathlib import Path


def main():
    with Path("results/ovo_simplestream_fullset/qwen3vl8b_nf4_fast_hld_full_t4.json").open(encoding="utf-8") as f:
        r = json.load(f)
    preds = [p for p in r.get("predictions", []) if p.get("task_type") == "HLD"]
    n = len(preds)
    print(f"HLD predictions: {n}")
    print()

    # Letter distribution
    gt_letters = []
    pred_letters = []
    for p in preds:
        gt_letters.append(chr(65 + int(p["ground_truth"])))
        if isinstance(p.get("predicted"), int) and 0 <= p["predicted"] < 4:
            pred_letters.append(chr(65 + p["predicted"]))
        else:
            pred_letters.append("?")
    print("Ground truth letter distribution:", dict(Counter(gt_letters)))
    print("Model prediction letter distribution:", dict(Counter(pred_letters)))
    print()

    # Confusion (gt vs pred)
    print("Confusion (rows=gt, cols=pred):")
    letters = ["A", "B", "C", "D", "?"]
    print(f"        {'  '.join(letters)}")
    for gtl in ["A", "B", "C", "D"]:
        row = []
        for prl in letters:
            row.append(sum(1 for p in preds if chr(65 + int(p["ground_truth"])) == gtl
                           and (("?" if not isinstance(p.get("predicted"), int) else chr(65 + p["predicted"])) == prl)))
        print(f"  gt={gtl}  " + "  ".join(f"{c:>2d}" for c in row))
    print()

    # Length of raw answer_text
    lens = [len(p.get("answer_text") or "") for p in preds]
    print(f"answer_text length: min={min(lens)} median={sorted(lens)[len(lens)//2]} max={max(lens)}")
    short = sum(1 for l in lens if l <= 3)
    print(f"  responses <= 3 chars: {short}/{n} ({100*short/n:.1f}%)")
    print()

    # Show 10 sample records
    print("Sample records (first 10):")
    for p in preds[:10]:
        gt = chr(65 + int(p["ground_truth"]))
        pred = p.get("predicted")
        pred_letter = chr(65 + pred) if isinstance(pred, int) and 0 <= pred < 4 else "?"
        ans = (p.get("answer_text") or "").replace("\n", " ")
        if len(ans) > 80:
            ans = ans[:77] + "..."
        mark = "[OK]" if p.get("correct") else "[XX]"
        q = (p.get("question") or "").replace("\n", " ")
        if len(q) > 60:
            q = q[:57] + "..."
        opts = p.get("options") or []
        print(f"  {mark} id={p['video_id']:>4} gt={gt} pred={pred_letter}  ans={ans!r}")
        print(f"        q={q!r}")
        print(f"        opts={opts}")


if __name__ == "__main__":
    main()
