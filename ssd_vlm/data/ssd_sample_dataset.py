"""
SSD sample dataset for multimodal LoRA fine-tuning.

By default this replays the original video frames used during SSD sample
generation so the student is trained as a VLM, not as a text-only student.
"""

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from ssd_vlm.data.video_utils import load_video_frames, resolve_video_path

logger = logging.getLogger(__name__)


class SSDSampleDataset(Dataset):
    """
    Dataset for SSD-generated samples stored as JSONL.

    Supports two modes:
    - multimodal replay with a processor (default for actual training)
    - legacy text-only tokenization with a tokenizer (kept for lightweight tests)
    """

    def __init__(
        self,
        samples_path: str,
        processor=None,
        tokenizer=None,
        source_data_path: Optional[str] = None,
        num_frames: int = 4,
        frame_sampling_strategy: str = "uniform",
        resize_shortest_edge: int = 224,
        max_seq_length: int = 4096,
        vqa_buffer_ratio: float = 0.1,
        cache_dir: Optional[str] = None,
        enable_cache: bool = True,
        seed: int = 42,
    ):
        self.samples_path = Path(samples_path)
        self.processor = processor
        self.tokenizer = tokenizer
        self.source_data_path = source_data_path
        self.num_frames = num_frames
        self.frame_sampling_strategy = frame_sampling_strategy
        self.resize_shortest_edge = resize_shortest_edge
        self.max_seq_length = max_seq_length
        self.vqa_buffer_ratio = vqa_buffer_ratio
        self.enable_cache = enable_cache
        self.cache_dir = Path(cache_dir) if cache_dir else None

        if self.processor is None and self.tokenizer is None:
            raise ValueError("Either processor or tokenizer must be provided")

        self.raw_samples = self._load_samples(samples_path)

        rng = random.Random(seed)
        n_vqa = int(len(self.raw_samples) * vqa_buffer_ratio)
        self.vqa_indices = set(rng.sample(range(len(self.raw_samples)), n_vqa)) if n_vqa else set()

        logger.info(
            "Loaded %d SSD samples (%d open-ended VQA, %d MC)",
            len(self.raw_samples),
            n_vqa,
            len(self.raw_samples) - n_vqa,
        )

    @staticmethod
    def _load_samples(path: str) -> List[Dict[str, Any]]:
        samples = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples

    def __len__(self) -> int:
        return len(self.raw_samples)

    def _format_mc(self, sample: Dict[str, Any]) -> str:
        options_text = "\n".join(
            f"{chr(65 + i)}: {opt}" for i, opt in enumerate(sample.get("options", []))
        )
        return (
            f"Question: {sample['question']}\n\n"
            f"Options:\n{options_text}\n\n"
            f"Answer:"
        )

    def _format_vqa(self, sample: Dict[str, Any]) -> str:
        return f"Question: {sample['question']}\n\nAnswer:"

    def _resolve_video(self, sample: Dict[str, Any]) -> Tuple[Path, Path]:
        root = self.source_data_path or sample.get("source_data_path")
        if not root:
            raise ValueError(
                "SSD sample is missing source_data_path and no source_data_path override was provided"
            )
        data_root = Path(root)
        video_path = resolve_video_path(
            data_path=data_root,
            video_id=sample["video_id"],
            video_relpath=sample.get("video_relpath"),
        )
        return data_root, video_path

    def _load_multimodal_item(self, idx: int) -> Dict[str, Any]:
        sample = self.raw_samples[idx]
        prompt = self._format_vqa(sample) if idx in self.vqa_indices else self._format_mc(sample)
        completion = sample.get("completion", "").strip()
        data_root, video_path = self._resolve_video(sample)
        cache_dir = self.cache_dir or (data_root / ".frame_cache")
        frames, frame_indices, total_frames = load_video_frames(
            video_path=video_path,
            num_frames=sample.get("num_frames", self.num_frames),
            frame_sampling_strategy=sample.get(
                "frame_sampling_strategy",
                self.frame_sampling_strategy,
            ),
            resize_shortest_edge=self.resize_shortest_edge,
            cache_dir=cache_dir,
            enable_cache=self.enable_cache,
            frame_indices=sample.get("frame_indices"),
        )
        return {
            "frames": frames,
            "prompt": prompt,
            "completion": completion,
            "video_id": sample["video_id"],
            "task_type": sample.get("task_type", ""),
            "skill_category": sample.get("skill_category", ""),
            "answer_idx": sample.get("answer_idx", 0),
            "video_path": str(video_path),
            "frame_indices": frame_indices,
            "total_frames": total_frames,
        }

    def _load_text_only_item(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.raw_samples[idx]
        prompt = self._format_vqa(sample) if idx in self.vqa_indices else self._format_mc(sample)
        completion = sample.get("completion", "")
        full_text = prompt + " " + completion

        encoding = self.tokenizer(
            full_text,
            max_length=self.max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        prompt_encoding = self.tokenizer(
            prompt,
            max_length=self.max_seq_length,
            truncation=True,
            return_tensors="pt",
        )
        prompt_len = prompt_encoding["input_ids"].shape[1]

        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.processor is not None:
            return self._load_multimodal_item(idx)
        return self._load_text_only_item(idx)


class SSDSampleDataCollator:
    """Collator for either multimodal replay batches or legacy text-only batches."""

    def __init__(self, processor=None, tokenizer=None, max_seq_length: int = 4096):
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def _collate_text_only(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": torch.stack([f["input_ids"] for f in features]),
            "attention_mask": torch.stack([f["attention_mask"] for f in features]),
            "labels": torch.stack([f["labels"] for f in features]),
        }

    def _collate_multimodal(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        full_conversations = []
        prompt_conversations = []

        for feature in features:
            user_content = [
                {"type": "image", "image": feature["frames"]},
                {"type": "text", "text": feature["prompt"]},
            ]
            full_conversations.append([
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": [{"type": "text", "text": feature["completion"]}]},
            ])
            prompt_conversations.append([
                {"role": "user", "content": user_content},
            ])

        full_inputs = self.processor.apply_chat_template(
            full_conversations,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={
                "padding": True,
                "truncation": True,
                "max_length": self.max_seq_length,
            },
        )
        prompt_inputs = self.processor.apply_chat_template(
            prompt_conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={
                "padding": True,
                "truncation": True,
                "max_length": self.max_seq_length,
            },
        )

        full_inputs.pop("token_type_ids", None)
        prompt_inputs.pop("token_type_ids", None)

        labels = full_inputs["input_ids"].clone()
        labels[full_inputs["attention_mask"] == 0] = -100
        for row_idx in range(labels.size(0)):
            prompt_len = int(prompt_inputs["attention_mask"][row_idx].sum().item())
            labels[row_idx, :min(prompt_len, labels.size(1))] = -100

        batch = dict(full_inputs)
        batch["labels"] = labels
        return batch

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if self.processor is not None:
            return self._collate_multimodal(features)
        return self._collate_text_only(features)


def _make_loader(
    dataset: Dataset,
    collator: SSDSampleDataCollator,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    pin_memory: bool,
    drop_last: bool,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
) -> DataLoader:
    use_workers = num_workers > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=collator,
        drop_last=drop_last and len(dataset) >= batch_size,
        persistent_workers=persistent_workers if use_workers else False,
        prefetch_factor=prefetch_factor if use_workers else None,
    )


def create_ssd_sample_dataloaders(
    samples_path: str,
    processor=None,
    tokenizer=None,
    source_data_path: Optional[str] = None,
    batch_size: int = 2,
    eval_batch_size: Optional[int] = None,
    num_workers: int = 4,
    max_seq_length: int = 4096,
    num_frames: int = 4,
    frame_sampling_strategy: str = "uniform",
    resize_shortest_edge: int = 224,
    vqa_buffer_ratio: float = 0.1,
    validation_split_ratio: float = 0.0,
    pin_memory: bool = True,
    drop_last: bool = False,
    enable_cache: bool = True,
    seed: int = 42,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    dataset = SSDSampleDataset(
        samples_path=samples_path,
        processor=processor,
        tokenizer=tokenizer,
        source_data_path=source_data_path,
        num_frames=num_frames,
        frame_sampling_strategy=frame_sampling_strategy,
        resize_shortest_edge=resize_shortest_edge,
        max_seq_length=max_seq_length,
        vqa_buffer_ratio=vqa_buffer_ratio,
        enable_cache=enable_cache,
        seed=seed,
    )
    collator = SSDSampleDataCollator(
        processor=processor,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )

    eval_batch_size = eval_batch_size or batch_size
    eval_dataloader = None

    if validation_split_ratio > 0 and len(dataset) > 1:
        eval_size = max(1, int(len(dataset) * validation_split_ratio))
        if eval_size >= len(dataset):
            eval_size = 1
        train_size = len(dataset) - eval_size
        generator = torch.Generator().manual_seed(seed)
        train_dataset, eval_dataset = random_split(
            dataset,
            [train_size, eval_size],
            generator=generator,
        )
        train_dataloader = _make_loader(
            dataset=train_dataset,
            collator=collator,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )
        eval_dataloader = _make_loader(
            dataset=eval_dataset,
            collator=collator,
            batch_size=eval_batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=pin_memory,
            drop_last=False,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )
    else:
        train_dataloader = _make_loader(
            dataset=dataset,
            collator=collator,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )

    return train_dataloader, eval_dataloader


def create_ssd_sample_dataloader(**kwargs) -> DataLoader:
    """Backward-compatible helper that returns only the training loader."""
    train_dataloader, _ = create_ssd_sample_dataloaders(**kwargs)
    return train_dataloader
