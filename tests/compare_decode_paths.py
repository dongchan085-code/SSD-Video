"""Diff our _fetch_simplestream_frames against SimpleStream's reference decode.

Hypothesis: our path uses qwen_exact_recent_decoder when
``chunk_duration * fps == 1.0`` automatically, while the published
SimpleStream run only enables it when ``QWEN_EXACT_RECENT_DECODE=1`` is set
in the environment. The released ``run_qwen3vl_ovo_4gpu.sh`` does NOT set
that env var — so the paper baseline runs through ``fetch_video`` + bucket-
by-time + last-N-chunks. If the two paths pick different frame indices for
HLD's long videos, that explains the -15pp gap with no model-side change.

This script picks 3 HLD video ids from the existing 100% partial file and
runs both decode paths against the same files. Prints frame indices,
timestamps, chunk ids, and a unified diff.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Allow importing SimpleStream's reference lib from the local snapshot.
SS_REF = Path(os.environ.get("SS_REF_DIR", r"C:/Users/swsuser-j07/AppData/Local/Temp/ss_ref"))
if str(SS_REF) not in sys.path:
    sys.path.insert(0, str(SS_REF))

# Force the env var OFF so we mimic the published baseline path.
os.environ.pop("QWEN_EXACT_RECENT_DECODE", None)

from ssd_vlm.data.video_utils import _fetch_simplestream_frames  # noqa: E402


def load_simplestream_decode():
    from lib.recent_window_eval import decode_video_to_chunks_qwen

    return decode_video_to_chunks_qwen


def pick_hld_videos(partial_path: Path, chunked_dir: Path, n: int = 3) -> list[Path]:
    seen = []
    with partial_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("task_type") != "HLD":
                continue
            vid = str(r["video_id"])
            p = chunked_dir / f"{vid}.mp4"
            if p.exists():
                seen.append(p)
            if len(seen) >= n:
                break
    return seen


def main() -> None:
    partial = Path("results/ovo_simplestream_fullset/qwen3vl8b_nf4_fast_hld_full_t4.partial_predictions.jsonl")
    chunked_dir = Path("D:/ssd_video_data/chunked_videos")
    videos = pick_hld_videos(partial, chunked_dir, n=3)
    if not videos:
        raise SystemExit("no HLD videos found")

    decode_video_to_chunks_qwen = load_simplestream_decode()

    for video_path in videos:
        print("=" * 70)
        print(f"VIDEO: {video_path.name}")
        print("=" * 70)

        # Path A: SimpleStream reference (fetch_video + bucket, no exact-recent)
        chunks, backend = decode_video_to_chunks_qwen(
            video_path=str(video_path),
            chunk_duration=1.0,
            fps=1.0,
            recent_frames_only=4,
        )
        recent = chunks[-4:]
        ss_indices = []
        ss_ts = []
        ss_chunk_ids = []
        for c in recent:
            # EvalChunk doesn't carry raw frame_idx, but timestamps come from it
            ss_ts.extend(c.frame_timestamps)
            ss_chunk_ids.extend([c.chunk_index] * len(c.frames))
            ss_indices.extend([int(round(t)) for t in c.frame_timestamps])  # at fps=1.0, idx ≈ ts
        print(f"\n[SimpleStream reference] backend={backend}")
        print(f"  total_chunks={len(chunks)}, recent kept={len(recent)}")
        print(f"  timestamps={ss_ts}")
        print(f"  chunk_ids={ss_chunk_ids}")
        print(f"  approx_indices={ss_indices}")
        print(f"  n_frames={sum(len(c.frames) for c in recent)}")

        # Path B: our _fetch_simplestream_frames (auto exact-recent for cd*fps==1)
        pil, indices, total, ts, chunk_ids = _fetch_simplestream_frames(
            video_path=video_path,
            chunk_duration=1.0,
            fps=1.0,
            recent_frames_only=4,
            resize_shortest_edge=None,
        )
        print(f"\n[Our path]")
        print(f"  total_frames_from_decoder={total}")
        print(f"  timestamps={ts}")
        print(f"  chunk_ids={chunk_ids}")
        print(f"  indices={indices}")
        print(f"  n_frames={len(pil)}")

        same_indices = sorted(ss_indices) == sorted(indices)
        same_ts = sorted([round(x, 3) for x in ss_ts]) == sorted([round(x, 3) for x in ts])
        print(f"\n  --> same approx_indices? {same_indices}")
        print(f"  --> same timestamps?     {same_ts}")
        print()


if __name__ == "__main__":
    main()
