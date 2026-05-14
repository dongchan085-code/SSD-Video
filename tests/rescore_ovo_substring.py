"""Rescore our HLD@100% predictions with OVO-Bench's official substring scoring.

Hypothesis (the actual one): SimpleStream paper headline numbers come from
``scoring/score_ovo_bench.py`` which inlines OVO-Bench official
``OVOBenchOfflineScore.calculate_score_backward_realtime``:

    def get_score(response, gt):
        if response is None: return 0
        return int(gt in response)

This is a substring match on the raw answer text (case-sensitive). Our code
in ssd_vlm/simplestream.extract_choice() uses a regex ``\\b([A-D])\\b``
first-match, which discards a correct letter that the model emits past the
first capital letter. On HLD where the model often writes a phrase before
the letter (e.g., "...involves Object D, while Object A..."), substring vs
first-letter-regex differ frequently.

Rescoring uses our existing answer_text without re-running inference.
"""

import json
from pathlib import Path

INPUTS = [
    ("HLD 100% (fullset)", Path("results/ovo_simplestream_fullset/qwen3vl8b_nf4_fast_hld_full_t4.json")),
    ("25% sample (snapshot)", Path("results/ovo_simplestream_fullset/qwen3vl8b_nf4_fast_25pct_t4.snapshot.json")),
    ("25% + HLD@100% combined", Path("results/ovo_simplestream_fullset/qwen3vl8b_nf4_fast_25pct_plus_hld_full_t4.json")),
]


def substring_score(response, gt_letter):
    if response is None or not str(response).strip():
        return 0
    return int(gt_letter in str(response))


def regex_correct_field(p) -> int:
    return 1 if p.get("correct") else 0


def main():
    for label, path in INPUTS:
        if not path.exists():
            print(f"[skip] {label}: not found at {path}")
            continue
        with path.open(encoding="utf-8") as f:
            r = json.load(f)
        preds = r.get("predictions", [])

        from collections import defaultdict
        by_task_regex = defaultdict(lambda: [0, 0])
        by_task_substr = defaultdict(lambda: [0, 0])

        for p in preds:
            task = p.get("task_type", "?")
            n_regex = regex_correct_field(p)
            response = p.get("answer_text")
            gt = p.get("ground_truth")

            # for multi-choice tasks: ground_truth is 0..3 int (our normalize), turn to letter
            if task in {"OCR", "ACR", "ATR", "STU", "FPD", "OJR", "EPM", "ASI", "HLD"}:
                try:
                    gt_letter = chr(65 + int(gt))
                except (TypeError, ValueError):
                    gt_letter = str(gt).upper() if gt is not None else ""
                n_substr = substring_score(response, gt_letter)
            else:
                # forward tasks: keep our regex/yesno field as-is, substring NA
                n_substr = n_regex

            by_task_regex[task][0] += n_regex
            by_task_regex[task][1] += 1
            by_task_substr[task][0] += n_substr
            by_task_substr[task][1] += 1

        print("=" * 70)
        print(f"{label}    [{path.name}]")
        print("=" * 70)
        print(f"{'task':<5} {'n':>4} {'regex%':>8} {'substr%':>9} {'Δ pp':>7}")
        print("-" * 38)
        tasks_order = ["OCR", "ACR", "ATR", "STU", "FPD", "OJR", "EPM", "ASI", "HLD", "REC", "SSR", "CRR"]
        for task in tasks_order:
            if task not in by_task_regex:
                continue
            c_r, n = by_task_regex[task]
            c_s, _ = by_task_substr[task]
            r_pct = 100.0 * c_r / n
            s_pct = 100.0 * c_s / n
            print(f"{task:<5} {n:>4} {r_pct:>7.2f}% {s_pct:>8.2f}% {s_pct - r_pct:>+7.2f}")
        print()


if __name__ == "__main__":
    main()
