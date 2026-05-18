import json
import tempfile
from pathlib import Path
import unittest

from scripts.diagnose_hld_repro import (
    audit_annotations,
    compare_scoring,
    _load_cache_record,
)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


class TestDiagnoseHldRepro(unittest.TestCase):
    def test_audit_annotations_flags_five_option_hld_and_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            anno = root / "ovo.json"
            manifest = root / "manifest.json"
            cache = root / "cache"

            annotations = [
                {
                    "id": "471",
                    "task": "HLD",
                    "question": "Where is it?",
                    "options": ["A opt", "B opt", "C opt", "D opt", "E opt"],
                    "gt": 4,
                },
                {
                    "id": "472",
                    "task": "HLD",
                    "question": "What happened?",
                    "options": ["A opt", "B opt", "C opt", "D opt"],
                    "gt": 2,
                },
            ]
            _write_json(anno, annotations)
            _write_json(manifest, annotations)

            for video_id in ("471", "472"):
                frame_dir = cache / video_id
                frame_dir.mkdir(parents=True, exist_ok=True)
                frames = []
                for i in range(8):
                    name = f"{i:04d}.png"
                    frames.append({
                        "file": name,
                        "frame_index": 100 + i,
                        "timestamp": float(i),
                        "chunk_id": i,
                    })
                _write_json(
                    frame_dir / "metadata.json",
                    {
                        "video_id": video_id,
                        "task_type": "HLD",
                        "recent_frames_only": 8,
                        "chunk_duration": 1.0,
                        "fps": 1.0,
                        "frames": frames,
                    },
                )

            summary, rows = audit_annotations(
                anno,
                task="HLD",
                manifest_path=manifest,
                cache_dir=cache,
                recent_frames=4,
            )

            self.assertEqual(summary["n_annotations"], 2)
            self.assertEqual(summary["non4_options"], 1)
            self.assertEqual(summary["gt_outside_abcd"], 1)
            self.assertEqual(summary["cache_missing"], 0)
            self.assertEqual(summary["cache_lt_recent"], 0)
            self.assertEqual(rows[0]["cache_indices"], [104, 105, 106, 107])

    def test_score_compare_separates_regex_from_official_substring(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = Path(tmp) / "result.json"
            _write_json(
                result,
                {
                    "predictions": [
                        {
                            "video_id": "1",
                            "task_type": "HLD",
                            "ground_truth": 2,
                            "answer_text": "C",
                            "correct": True,
                        },
                        {
                            "video_id": "2",
                            "task_type": "HLD",
                            "ground_truth": 4,
                            "answer_text": "E",
                            "correct": False,
                        },
                    ]
                },
            )

            summary, rows = compare_scoring(result, task="HLD")

            hld = summary["tasks"]["HLD"]
            self.assertEqual(hld["n"], 2)
            self.assertEqual(hld["release_regex_percent"], 50.0)
            self.assertEqual(hld["official_substring_percent"], 100.0)
            self.assertEqual(summary["num_regex_substring_diffs"], 1)
            self.assertTrue(rows[1]["regex_vs_substring_diff"])

    def test_load_cache_record_supports_precomputed_meta(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame_dir = root / "vid"
            frame_dir.mkdir(parents=True, exist_ok=True)
            _write_json(
                frame_dir / "meta.json",
                {
                    "frame_indices": list(range(8)),
                    "frame_timestamps": [float(i) for i in range(8)],
                    "chunk_duration": 1.0,
                    "saved_count": 8,
                },
            )

            record = _load_cache_record(root, "vid", recent=4)

            self.assertEqual(record["cache_format"], "precomputed_frames")
            self.assertEqual(record["cache_indices"], [4, 5, 6, 7])
            self.assertEqual(record["cache_chunk_ids"], [4, 5, 6, 7])


if __name__ == "__main__":
    unittest.main()
