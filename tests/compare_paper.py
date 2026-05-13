"""Compare our 10% subset Qwen3-VL-8B NF4 numbers against OVO-Bench paper Table 1.

Paper numbers transcribed from arXiv 2501.05510v2 / JoeLeelyf/ovo-bench README.
Our scoring: per-task accuracy as fraction. Paper reports percentage; we
multiply ours by 100 to compare.

The paper computes Backward Avg / Realtime Avg / Forward Avg / Total Avg as
the macro-average of per-task accuracies within each category. We replicate
that here so the numbers are directly comparable.
"""

import json
from collections import defaultdict


BACKWARD = ["EPM", "ASI", "HLD"]
REALTIME = ["OCR", "ACR", "ATR", "STU", "FPD", "OJR"]
FORWARD = ["REC", "SSR", "CRR"]


PAPER = {
    "Human":            {"EPM":92.59,"ASI":93.02,"HLD":91.37,"OCR":93.96,"ACR":92.57,"ATR":94.83,"STU":92.70,"FPD":91.09,"OJR":94.02,"REC":95.48,"SSR":89.67,"CRR":93.56},
    "Gemini-1.5-Pro":   {"EPM":58.59,"ASI":76.35,"HLD":52.64,"OCR":85.91,"ACR":66.97,"ATR":79.31,"STU":58.43,"FPD":63.37,"OJR":61.96,"REC":35.53,"SSR":74.24,"CRR":61.67},
    "GPT-4o":           {"EPM":57.91,"ASI":75.68,"HLD":48.66,"OCR":69.80,"ACR":64.22,"ATR":71.55,"STU":51.12,"FPD":70.30,"OJR":59.78,"REC":27.58,"SSR":73.21,"CRR":59.40},
    "Qwen2-VL-72B":     {"EPM":52.53,"ASI":60.81,"HLD":57.53,"OCR":65.77,"ACR":60.55,"ATR":69.83,"STU":51.69,"FPD":69.31,"OJR":54.35,"REC":38.83,"SSR":64.07,"CRR":45.00},
    "Qwen2-VL-7B":      {"EPM":47.81,"ASI":35.48,"HLD":56.08,"OCR":60.40,"ACR":50.46,"ATR":56.03,"STU":47.19,"FPD":66.34,"OJR":55.43,"REC":31.66,"SSR":65.82,"CRR":48.75},
    "LLaVA-Video-7B":   {"EPM":56.23,"ASI":57.43,"HLD": 7.53,"OCR":69.13,"ACR":58.72,"ATR":68.83,"STU":49.44,"FPD":74.26,"OJR":59.78,"REC":34.10,"SSR":69.95,"CRR":60.42},
    "LLaVA-OneVision-7B":{"EPM":54.21,"ASI":55.41,"HLD":21.51,"OCR":66.44,"ACR":57.80,"ATR":73.28,"STU":53.37,"FPD":71.29,"OJR":61.96,"REC":25.64,"SSR":67.09,"CRR":58.75},
    "InternVL2-8B":     {"EPM":48.15,"ASI":57.43,"HLD":24.73,"OCR":67.11,"ACR":60.55,"ATR":63.79,"STU":46.07,"FPD":68.32,"OJR":56.52,"REC":26.50,"SSR":59.14,"CRR":54.14},
    "LongVU-7B":        {"EPM":40.74,"ASI":59.46,"HLD": 4.84,"OCR":53.69,"ACR":53.21,"ATR":62.93,"STU":47.75,"FPD":68.32,"OJR":59.78,"REC":12.18,"SSR":69.48,"CRR":60.83},
}


def category_macro(scores, tasks):
    vals = [scores[t] for t in tasks if t in scores]
    return sum(vals) / len(vals) if vals else None


def per_task_from_predictions(r):
    by_task = defaultdict(lambda: [0, 0])
    for p in r.get("predictions", []):
        t = p.get("task_type")
        by_task[t][1] += 1
        if p.get("correct"):
            by_task[t][0] += 1
    return {t: 100.0 * c / n for t, (c, n) in by_task.items() if n}


def main():
    with open("results/ovo_10pct/qwen3vl8b_nf4_t4.json", encoding="utf-8") as f:
        r = json.load(f)
    ours_label = "Qwen3VL-8B-NF4 (10% T4)"
    ours_pct = per_task_from_predictions(r)
    ours_n = defaultdict(int)
    for p in r.get("predictions", []):
        ours_n[p.get("task_type")] += 1

    rows = list(PAPER.items()) + [(ours_label, ours_pct)]
    all_tasks = BACKWARD + REALTIME + FORWARD

    print("Per-task accuracy (%) — paper Table 1 vs ours")
    print("=" * 140)
    header = ["model"] + all_tasks + ["B.Avg", "R.Avg", "F.Avg", "Tot"]
    print(" ".join(f"{h:>7}" for h in [header[0]] + header[1:]))
    print("-" * 140)
    for label, scores in rows:
        bw = category_macro(scores, BACKWARD)
        rt = category_macro(scores, REALTIME)
        fw = category_macro(scores, FORWARD)
        tot = (bw + rt + fw) / 3 if None not in (bw, rt, fw) else None
        cells = [label[:22].ljust(22)]
        for t in all_tasks:
            v = scores.get(t)
            cells.append(f"{v:5.1f}" if v is not None else "  -  ")
        for v in (bw, rt, fw, tot):
            cells.append(f"{v:5.1f}" if v is not None else "  -  ")
        print(" ".join(c for c in cells))

    print()
    print("Δ vs Qwen2-VL-7B (paper) on the same tasks (ours - paper, percentage points):")
    print("-" * 140)
    paper = PAPER["Qwen2-VL-7B"]
    bw_p, rt_p, fw_p = category_macro(paper, BACKWARD), category_macro(paper, REALTIME), category_macro(paper, FORWARD)
    bw_o, rt_o, fw_o = category_macro(ours_pct, BACKWARD), category_macro(ours_pct, REALTIME), category_macro(ours_pct, FORWARD)
    for t in all_tasks:
        op = ours_pct.get(t)
        pp = paper.get(t)
        if op is None or pp is None:
            continue
        n = ours_n.get(t, 0)
        delta = op - pp
        print(f"  {t:4s}  ours={op:5.1f}  paper={pp:5.1f}  Δ={delta:+6.1f}  (our n={n})")
    print(f"  Backward macro avg:  ours={bw_o:5.1f}  paper={bw_p:5.1f}  Δ={bw_o - bw_p:+6.1f}")
    print(f"  Realtime macro avg:  ours={rt_o:5.1f}  paper={rt_p:5.1f}  Δ={rt_o - rt_p:+6.1f}")
    print(f"  Forward  macro avg:  ours={fw_o:5.1f}  paper={fw_p:5.1f}  Δ={fw_o - fw_p:+6.1f}")
    tot_p = (bw_p + rt_p + fw_p) / 3
    tot_o = (bw_o + rt_o + fw_o) / 3
    print(f"  Total avg:          ours={tot_o:5.1f}  paper={tot_p:5.1f}  Δ={tot_o - tot_p:+6.1f}")


if __name__ == "__main__":
    main()
