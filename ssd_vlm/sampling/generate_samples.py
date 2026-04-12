"""
Generate SSD samples from frozen Qwen3-VL-8B-Instruct model.

High-temperature (1.5) sampling with top-k (10) to create diverse training data.
No filtering, no verification - raw model outputs for label-free fine-tuning.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from ssd_vlm.data.perception_test_dataset import PerceptionTestDataset

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
    ):
        """
        Initialize sample generator.
        
        Args:
            model_id: HuggingFace model ID
            dtype: Data type (bfloat16, float16, float32)
            device_map: Device mapping for model
            temperature: Sampling temperature
            top_k: Top-k tokens to consider
            top_p: Top-p (nucleus) sampling
            max_new_tokens: Maximum tokens to generate
            batch_size: Batch size for inference
        """
        self.model_id = model_id
        self.dtype = dtype
        self.device_map = device_map
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        
        # Setup dtype
        if dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16
        elif dtype == "float16":
            self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch.float32
        
        # Load model and processor
        logger.info(f"Loading model: {model_id}")
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.model.eval()
        logger.info("Model loaded successfully")
    
    def _format_prompt(
        self,
        question: str,
        options: List[str],
    ) -> str:
        """Format question and options into prompt."""
        options_text = "\n".join(
            f"{chr(65 + i)}: {opt}" for i, opt in enumerate(options)
        )
        prompt = f"""Question: {question}

Options:
{options_text}

Answer:"""
        return prompt
    
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
        
        # Open output file
        output_file = open(output_path, 'w')
        
        try:
            sample_count = 0
            
            with tqdm(total=samples_to_generate, desc="Generating samples") as pbar:
                for idx in range(samples_to_generate):
                    sample = dataset[idx]
                    
                    # Format prompt
                    prompt = self._format_prompt(
                        question=sample["question"],
                        options=sample["options"],
                    )
                    
                    # Prepare inputs
                    # Note: This is simplified - actual implementation would handle images properly
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": sample["frames"],  # [T, 3, H, W]
                                },
                                {
                                    "type": "text",
                                    "text": prompt,
                                }
                            ]
                        }
                    ]
                    
                    # Process input — unified apply_chat_template (Qwen3-VL API)
                    inputs = self.processor.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=True,
                        return_tensors="pt",
                    )
                    inputs.pop("token_type_ids", None)
                    
                    # Move inputs to device
                    inputs = {k: v.to(self.model.device) if torch.is_tensor(v) else v 
                             for k, v in inputs.items()}
                    
                    # Generate
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                        top_k=self.top_k,
                        top_p=self.top_p,
                        do_sample=True,
                        use_cache=True,
                    )
                    
                    # Decode
                    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
                    completion = self.processor.decode(
                        generated_ids,
                        skip_special_tokens=True,
                    )
                    
                    # Build output sample
                    output_sample = {
                        "video_id": sample["video_id"],
                        "question": sample["question"],
                        "options": sample["options"],
                        "answer_idx": sample["answer_idx"],
                        "skill_category": sample["skill_category"],
                        "task_type": sample["task_type"],
                        "completion": completion,
                        "completion_tokens": len(generated_ids),
                        "temperature": self.temperature,
                        "top_k": self.top_k,
                    }
                    
                    # Write to file
                    output_file.write(json.dumps(output_sample) + "\n")
                    
                    sample_count += 1
                    pbar.update(1)
                    
                    # Periodic save (file is already being written)
                    if sample_count % save_interval == 0:
                        output_file.flush()
                        logger.info(f"Generated {sample_count} samples")
        
        finally:
            output_file.close()
        
        logger.info(f"Sample generation complete. Saved {sample_count} samples to {output_path}")
        return str(output_path)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


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
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load config
    config = load_config(args.config)
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
        enable_cache=True,
    )
    
    # Create generator
    generator = SSDSampleGenerator(
        model_id=config["model"].get("model_id", "Qwen/Qwen3-VL-8B-Instruct"),
        dtype=config["model"].get("dtype", "bfloat16"),
        device_map=config["model"].get("device_map", "auto"),
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
