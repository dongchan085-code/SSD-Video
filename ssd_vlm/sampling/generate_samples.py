"""
Generate SSD samples from frozen Qwen3-VL-8B-Instruct model.

High-temperature (1.5) sampling with top-k (10) to create diverse training data.
No filtering, no verification - raw model outputs for label-free fine-tuning.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ssd_vlm.data.perception_test_dataset import PerceptionTestDataset
from ssd_vlm.model_loading import load_vlm_processor_and_model
from ssd_vlm.simplestream import format_ovo_prompt
from ssd_vlm.utils.config import load_config

logger = logging.getLogger(__name__)


class SSDSampleGenerator:
    """Generate SSD samples from frozen model."""
    
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
        dtype: str = "bfloat16",
        device_map: str = "auto",
        temperature: float = 1.5,
        top_k: int = 10,
        top_p: float = 1.0,
        max_new_tokens: int = 512,
        batch_size: int = 32,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        attn_implementation: Optional[str] = None,
        max_pixels: Optional[int] = None,
        min_pixels: Optional[int] = None,
    ):
        """Initialize sample generator. See `load_vlm_processor_and_model` for quantization args."""
        self.model_id = model_id
        self.dtype = dtype
        self.device_map = device_map
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size

        logger.info(f"Loading model: {model_id}")
        self.processor, self.model = load_vlm_processor_and_model(
            model_path=model_id,
            dtype=dtype,
            device_map=device_map,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            attn_implementation=attn_implementation,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
        )
        self.model.eval()
        logger.info("Model loaded successfully")
    
    @staticmethod
    def _video_content(frames: Any) -> List[Dict[str, Any]]:
        # Send all frames as one video item so Qwen3-VL's temporal patching halves vision tokens.
        if isinstance(frames, list):
            return [{"type": "video", "video": frames}]
        return [{"type": "video", "video": [frames]}]
    
    @torch.no_grad()
    def generate_samples(
        self,
        dataset: PerceptionTestDataset,
        output_path: str,
        num_samples: Optional[int] = None,
        save_interval: int = 100,
    ) -> str:
        """
        Generate SSD samples from frozen model.
        
        Args:
            dataset: PerceptionTestDataset instance
            output_path: Path to save JSONL output
            num_samples: Number of samples to generate (None = all)
            save_interval: Save every N samples
        
        Returns:
            Path to output file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        num_samples = num_samples or len(dataset)
        samples_to_generate = min(num_samples, len(dataset))

        # Use a subset if needed
        if samples_to_generate < len(dataset):
            dataset = torch.utils.data.Subset(dataset, range(samples_to_generate))

        # DataLoader for async prefetch — workers decode next frames while GPU runs inference
        loader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=4,
            persistent_workers=True,
            prefetch_factor=8,
            pin_memory=True,
            shuffle=False,
            collate_fn=lambda batch: batch[0],
        )

        output_file = open(output_path, 'w')
        buffer: List[str] = []

        try:
            sample_count = 0

            for sample in tqdm(loader, total=len(loader), desc="Generating samples"):
                prompt = format_ovo_prompt(
                    task_type=sample.get("task_type", ""),
                    question=sample["question"],
                    options=sample["options"],
                )

                frame_images = sample.get("frame_images", sample["frames"])
                messages = [
                    {
                        "role": "user",
                        "content": self._video_content(frame_images)
                        + [{"type": "text", "text": prompt}],
                    }
                ]

                inputs = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                inputs.pop("token_type_ids", None)

                inputs = {k: v.to(self.model.device) if torch.is_tensor(v) else v
                         for k, v in inputs.items()}

                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    do_sample=True,
                    use_cache=True,
                )

                generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
                completion = self.processor.decode(
                    generated_ids,
                    skip_special_tokens=True,
                )

                output_sample = {
                    "video_id": sample["video_id"],
                    "question": sample["question"],
                    "options": sample["options"],
                    "answer_idx": sample["answer_idx"],
                    "skill_category": sample["skill_category"],
                    "task_type": sample["task_type"],
                    "completion": completion,
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": completion},
                    ],
                    "completion_tokens": len(generated_ids),
                    "temperature": self.temperature,
                    "top_k": self.top_k,
                    "top_p": self.top_p,
                    "source_data_path": sample.get("source_data_path"),
                    "source_split": sample.get("source_split"),
                    "video_relpath": sample.get("video_relpath"),
                    "frame_indices": sample.get("frame_indices"),
                    "frame_timestamps": sample.get("frame_timestamps"),
                    "chunk_ids": sample.get("chunk_ids"),
                    "total_frames": sample.get("total_frames"),
                    "num_frames": len(frame_images) if isinstance(frame_images, list) else int(sample["frames"].shape[0]),
                    "recent_frames_only": len(frame_images) if isinstance(frame_images, list) else int(sample["frames"].shape[0]),
                    "chunk_duration": sample.get("chunk_duration", 1.0),
                    "fps": sample.get("fps", 1.0),
                }

                buffer.append(json.dumps(output_sample))
                sample_count += 1

                if len(buffer) >= save_interval:
                    output_file.write("\n".join(buffer) + "\n")
                    output_file.flush()
                    buffer.clear()
                    logger.info(f"Generated {sample_count} samples")

            # Flush remaining
            if buffer:
                output_file.write("\n".join(buffer) + "\n")
                output_file.flush()

        finally:
            output_file.close()

        logger.info(f"Sample generation complete. Saved {sample_count} samples to {output_path}")
        return str(output_path)


def main():
    """Main script for generating SSD samples."""
    parser = argparse.ArgumentParser(description="Generate SSD samples from frozen model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="./outputs/ssd_samples",
                       help="Output directory")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Number of samples to generate (default: all)")
    parser.add_argument("--data_path", type=str, default="./data/perception_test",
                       help="Path to Perception Test data")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[],
        metavar="section.key=value",
        help="Override a config value, e.g. --set generation.temperature=1.2 (repeatable)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load config and apply any --set overrides
    config = load_config(args.config)
    for override in args.overrides:
        if "=" not in override:
            raise ValueError(f"--set override must be section.key=value, got: {override!r}")
        key_path, raw_val = override.split("=", 1)
        parts = key_path.split(".")
        node = config
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        # Try numeric coercion; fall back to string.
        try:
            node[parts[-1]] = int(raw_val)
        except ValueError:
            try:
                node[parts[-1]] = float(raw_val)
            except ValueError:
                node[parts[-1]] = raw_val
    logger.info(f"Loaded config from {args.config}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    logger.info(f"Loading Perception Test dataset from {args.data_path}")
    dataset = PerceptionTestDataset(
        data_path=args.data_path,
        split=config["data"].get("split", "train"),
        num_frames=config["data"].get("num_frames", 4),
        frame_sampling_strategy=config["data"].get("frame_sampling_strategy", "uniform"),
        resize_shortest_edge=config["data"].get("resize_shortest_edge", 224),
        memory_skill_oversample_ratio=config["data"].get("memory_skill_oversample_ratio", 2.0),
        recent_frames_only=config["data"].get("recent_frames_only"),
        chunk_duration=config["data"].get("chunk_duration", 1.0),
        fps=config["data"].get("fps", 1.0),
    )
    
    # Create generator
    generator = SSDSampleGenerator(
        model_id=config["model"].get("model_id", "Qwen/Qwen3-VL-8B-Instruct"),
        dtype=config["model"].get("dtype", "bfloat16"),
        device_map=config["model"].get("device_map", "auto"),
        load_in_8bit=config["model"].get("load_in_8bit", False),
        load_in_4bit=config["model"].get("load_in_4bit", False),
        attn_implementation=config["model"].get("attn_implementation"),
        max_pixels=config["model"].get("max_pixels"),
        min_pixels=config["model"].get("min_pixels"),
        temperature=config["generation"].get("temperature", 1.5),
        top_k=config["generation"].get("top_k", 10),
        top_p=config["generation"].get("top_p", 1.0),
        max_new_tokens=config["generation"].get("max_new_tokens", 512),
        batch_size=config["training"].get("batch_size", 32),
    )
    
    # Generate samples
    output_file = output_dir / "samples.jsonl"
    generator.generate_samples(
        dataset=dataset,
        output_path=str(output_file),
        num_samples=args.num_samples,
        save_interval=config["output"].get("save_interval", 100),
    )
    
    logger.info(f"Samples saved to {output_file}")


if __name__ == "__main__":
    main()
