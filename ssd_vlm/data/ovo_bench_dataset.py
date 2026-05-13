"""
OVO-Bench dataset utilities for real video-backed evaluation.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

from torch.utils.data import Dataset

from ssd_vlm.data.video_utils import (
    load_video_frames_dual,
    resolve_video_path,
)
from ssd_vlm.simplestream import (
    BACKWARD_TASK_SET,
    FORWARD_TASK_SET,
    REAL_TIME_TASK_SET,
    task_group,
)

logger = logging.getLogger(__name__)


LOCK_TASKS = REAL_TIME_TASK_SET
FORK_TASKS = BACKWARD_TASK_SET
REAL_TIME_TASKS = REAL_TIME_TASK_SET
BACKWARD_TASKS = BACKWARD_TASK_SET
FORWARD_TASKS = FORWARD_TASK_SET


class OVOBenchDataset(Dataset):
    """OVO-Bench dataset with on-demand frame loading."""

    def __init__(
        self,
        data_path: str,
        split: str = "test",
        num_frames: int = 4,
        frame_sampling_strategy: str = "uniform",
        resize_shortest_edge: int = 224,
        cache_dir: Optional[str] = None,
        enable_cache: bool = True,
        anno_path: Optional[str] = None,
        chunked_dir: Optional[str] = None,
        recent_frames_only: Optional[int] = None,
        chunk_duration: float = 1.0,
        fps: float = 1.0,
    ):
        self.data_path = Path(data_path)
        self.split = split
        self.num_frames = num_frames
        self.frame_sampling_strategy = frame_sampling_strategy
        self.resize_shortest_edge = resize_shortest_edge
        self.enable_cache = enable_cache
        self.recent_frames_only = recent_frames_only or num_frames
        self.chunk_duration = chunk_duration
        self.fps = fps
        self.chunked_dir = Path(chunked_dir) if chunked_dir else self.data_path / "chunked_videos"
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_path / ".frame_cache"
        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._leakage_suspect_re = re.compile(
            r'\b(currently|now|visible|present|this scene|in the frame|right now)\b',
            re.IGNORECASE,
        )
        self._pure_memory_re = re.compile(
            r'\b(before|earlier|previously|past|ago|happened)\b',
            re.IGNORECASE,
        )

        self.samples = []
        native_anno_path = Path(anno_path) if anno_path else self.data_path / "ovo_bench_new.json"
        if native_anno_path.exists():
            self._load_native_simplestream_annotations(native_anno_path)
        else:
            self._load_legacy_annotations(split)

        logger.info(f"Loaded {len(self.samples)} OVO-Bench {split} samples from {self.data_path}")

    def _chunked_relpath(self, filename: str) -> str:
        default_dir = self.data_path / "chunked_videos"
        if self.chunked_dir.resolve() == default_dir.resolve():
            return f"chunked_videos/{filename}"
        return str((self.chunked_dir / filename).resolve())

    def _is_pure_memory(self, question: str, task_type: str) -> bool:
        if task_type not in FORK_TASKS:
            return False
        has_leakage_cue = bool(self._leakage_suspect_re.search(question))
        has_memory_cue = bool(self._pure_memory_re.search(question))
        return has_memory_cue and not has_leakage_cue

    def _load_native_simplestream_annotations(self, anno_path: Path) -> None:
        with open(anno_path, "r", encoding="utf-8") as f:
            annotations = json.load(f)

        for annotation in annotations:
            task_type = annotation.get("task", annotation.get("task_type", ""))
            video_id = str(annotation.get("id", annotation.get("video_id", "")))
            if not video_id:
                continue
            question = annotation.get("question", "")
            options = annotation.get("options", [])

            if task_type in FORWARD_TASKS and isinstance(annotation.get("test_info"), list):
                for index, test_info in enumerate(annotation["test_info"]):
                    test_question = test_info.get("question", question)
                    test_options = test_info.get("options", options)
                    video_relpath = test_info.get(
                        "video_relpath",
                        self._chunked_relpath(f"{video_id}_{index}.mp4"),
                    )
                    self.samples.append({
                        "video_id": f"{video_id}_{index}",
                        "source_id": video_id,
                        "question": test_question,
                        "options": test_options,
                        "answer_idx": test_info.get("gt", test_info.get("answer_idx", 0)),
                        "task_type": task_type,
                        "video_relpath": video_relpath,
                        "pure_memory": False,
                        "ovo_split": task_group(task_type),
                    })
                continue

            video_relpath = annotation.get("video_relpath", self._chunked_relpath(f"{video_id}.mp4"))
            self.samples.append({
                "video_id": video_id,
                "source_id": video_id,
                "question": question,
                "options": options,
                "answer_idx": annotation.get("gt", annotation.get("answer_idx", 0)),
                "task_type": task_type,
                "video_relpath": video_relpath,
                "pure_memory": self._is_pure_memory(question, task_type),
                "ovo_split": task_group(task_type),
            })

    def _load_legacy_annotations(self, split: str) -> None:
        split_file = self.data_path / f"{split}_split.json"
        annotations_file = self.data_path / f"{split}_annotations.json"
        if not split_file.exists() or not annotations_file.exists():
            raise FileNotFoundError(
                f"OVO-Bench data not found. Expected {self.data_path / 'ovo_bench_new.json'} "
                f"or legacy {split_file.name}/{annotations_file.name}"
            )

        with open(split_file, "r", encoding="utf-8") as f:
            split_data = json.load(f)
        with open(annotations_file, "r", encoding="utf-8") as f:
            annotations = json.load(f)

        for video_id in split_data.get("video_ids", []):
            if video_id not in annotations:
                continue
            annotation = annotations[video_id]
            question = annotation.get("question", "")
            task_type = annotation.get("task_type", "")
            self.samples.append({
                "video_id": video_id,
                "source_id": video_id,
                "question": question,
                "options": annotation.get("options", []),
                "answer_idx": annotation.get("answer_idx", 0),
                "task_type": task_type,
                "video_relpath": annotation.get("video_relpath", f"videos/{video_id}.mp4"),
                "pure_memory": self._is_pure_memory(question, task_type),
                "ovo_split": task_group(task_type),
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx].copy()
        video_path = resolve_video_path(
            data_path=self.data_path,
            video_id=sample["video_id"],
            video_relpath=sample.get("video_relpath"),
        )
        frames, frame_images, frame_indices, total_frames, frame_timestamps, chunk_ids = (
            load_video_frames_dual(
                video_path=video_path,
                num_frames=self.num_frames,
                tensor_resize_shortest_edge=self.resize_shortest_edge,
                pil_resize_shortest_edge=self.resize_shortest_edge,
                frame_sampling_strategy=self.frame_sampling_strategy,
                cache_dir=self.cache_dir,
                enable_cache=self.enable_cache,
                recent_frames_only=self.recent_frames_only,
                chunk_duration=self.chunk_duration,
                fps=self.fps,
            )
        )
        sample.update({
            "frames": frames,
            "frame_images": frame_images,
            "frame_indices": frame_indices,
            "frame_timestamps": frame_timestamps,
            "chunk_ids": chunk_ids,
            "total_frames": total_frames,
            "video_path": str(video_path),
        })
        return sample
