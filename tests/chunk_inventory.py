"""Audit which OVO chunks have been written so far on D:\.

Reports per-task: (a) how many expected chunks already exist on disk, and
(b) how many SOURCE annotations have ALL their chunks present (so we can
tell which annotations are usable by the sample_ratio loader).
"""

import json
import os
from collections import defaultdict


FORWARD = {"REC", "SSR", "CRR"}
ANNO_PATH = "D:/ssd_video_data/ovo_bench_full.json"
CHUNKED_DIR = "D:/ssd_video_data/chunked_videos"


def main():
    with open(ANNO_PATH, encoding="utf-8") as f:
        anno = json.load(f)
    existing = set(os.listdir(CHUNKED_DIR))

    by_task_total = defaultdict(int)
    by_task_done = defaultdict(int)
    by_task_source = defaultdict(lambda: [0, 0])  # [complete_anno, total_anno]

    for a in anno:
        task = a["task"]
        vid = str(a["id"])
        if task in FORWARD:
            chunks = [f"{vid}_{i}.mp4" for i in range(len(a.get("test_info") or []))]
        else:
            chunks = [f"{vid}.mp4"]
        by_task_total[task] += len(chunks)
        done = sum(1 for c in chunks if c in existing)
        by_task_done[task] += done
        by_task_source[task][1] += 1
        if done == len(chunks):
            by_task_source[task][0] += 1

    total_done = sum(by_task_done.values())
    total_total = sum(by_task_total.values())
    print(f"Total chunks present: {total_done} / {total_total} ({100*total_done/total_total:.1f}%)")
    print()
    print("task  chunks_done/total   complete_anno/total")
    for t in sorted(by_task_total):
        cd, ct = by_task_done[t], by_task_total[t]
        sc, st = by_task_source[t]
        print(f"  {t:<3}  {cd:>5} / {ct:<5}  {sc:>4} / {st:<4}")


if __name__ == "__main__":
    main()
