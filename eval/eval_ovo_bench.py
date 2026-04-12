"""
OVO-Bench Evaluation for SSD-VLM.
Evaluates vision language models with 4-frame streaming budget.
Adapted from SimpleStream evaluation protocol.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import yaml
from tqdm import tqdm

from ssd_vlm.data.ovo_bench_dataset import FORK_TASKS, LOCK_TASKS, OVOBenchDataset
from ssd_vlm.model_loading import load_vlm_processor_and_model

logger = logging.getLogger(__name__)


class OVOBenchEvaluator:
    """Evaluator for OVO-Bench benchmark."""
    
    def __init__(
        self,
        model_path: str,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        num_frames: int = 4,
        frame_sampling_strategy: str = "uniform",
        resize_shortest_edge: int = 224,
        max_new_tokens: int = 512,
        batch_size: int = 16,
    ):
        """
        Initialize OVO-Bench evaluator.
        
        Args:
            model_path: Path to model (can be model ID or local path)
            dtype: Data type
            device_map: Device mapping
            num_frames: Number of frames (typically 4)
            max_new_tokens: Max generation tokens
            batch_size: Batch size for evaluation
        """
        self.model_path = model_path
        self.num_frames = num_frames
        self.frame_sampling_strategy = frame_sampling_strategy
        self.resize_shortest_edge = resize_shortest_edge
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size

        logger.info(f"Loading model from: {model_path}")
        self.processor, self.model = load_vlm_processor_and_model(
            model_path=model_path,
            dtype=dtype,
            device_map=device_map,
        )
        self.model.eval()
        logger.info("Model loaded successfully")
        
        # Task definitions
        # Temporal Lock tasks: real-time perception (sharp distributions needed)
        self.lock_tasks = LOCK_TASKS
        # Temporal Fork tasks: backward tracing / memory (flatter distributions needed)
        self.fork_tasks = FORK_TASKS
    
    def load_ovo_dataset(self, data_path: str, split: str = "test") -> OVOBenchDataset:
        """
        Load OVO-Bench dataset.
        
        Args:
            data_path: Path to OVO-Bench data directory
            split: Dataset split (test, val, etc.)
        
        Returns:
            List of test samples
        """
        return OVOBenchDataset(
            data_path=data_path,
            split=split,
            num_frames=self.num_frames,
            frame_sampling_strategy=self.frame_sampling_strategy,
            resize_shortest_edge=self.resize_shortest_edge,
        )
    
    def _format_prompt(self, question: str, options: List[str]) -> str:
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
    def _generate_answer(
        self,
        question: str,
        options: List[str],
        frames: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 1,
    ) -> str:
        """
        Generate answer for a single sample.
        
        Args:
            question: Question text
            options: List of options
            frames: [num_frames, 3, H, W] tensor
            temperature: Generation temperature
            top_k: Top-k sampling
        
        Returns:
            Generated text
        """
        prompt = self._format_prompt(question, options)
        
        # Prepare input
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": frames,
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ]
            }
        ]
        
        # Unified apply_chat_template (Qwen3-VL API)
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)

        # Move to device
        inputs = {k: v.to(self.model.device) if torch.is_tensor(v) else v
                 for k, v in inputs.items()}
        
        # Generate
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            do_sample=(temperature > 0),
            use_cache=True,
        )
        
        # Decode
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        answer = self.processor.decode(
            generated_ids,
            skip_special_tokens=True,
        )
        
        return answer.strip()
    
    def _extract_choice(self, text: str) -> Optional[int]:
        """
        Extract choice index from generated text.

        Priority order:
        1. Explicit "answer is X" / "answer: X" patterns
        2. Leading choice letter at start of text ("A.", "A)", "A ")
        3. Parenthesised choice "(A)"
        4. Bare word-boundary choice letter
        5. Digit (0-3)

        Args:
            text: Generated text

        Returns:
            Choice index (0-3) or None
        """
        import re

        text_stripped = text.strip()
        text_upper = text_stripped.upper()

        # 1. "answer is X" / "answer: X" / "answer = X"
        m = re.search(r'ANSWER\s*[IS:=]+\s*([A-D])\b', text_upper)
        if m:
            return ord(m.group(1)) - ord('A')

        # 2. Leading letter: "A.", "A)", "A " or just "A" at start
        m = re.match(r'^([A-D])[\.\)\s:]', text_upper)
        if m:
            return ord(m.group(1)) - ord('A')

        # 3. Parenthesised: "(A)" or "[A]"
        m = re.search(r'[\(\[]\s*([A-D])\s*[\)\]]', text_upper)
        if m:
            return ord(m.group(1)) - ord('A')

        # 4. Word-boundary standalone letter
        m = re.search(r'\b([A-D])\b', text_upper)
        if m:
            return ord(m.group(1)) - ord('A')

        # 5. Digit fallback
        m = re.search(r'\b([0-3])\b', text)
        if m:
            return int(m.group(1))

        return None
    
    def evaluate(
        self,
        samples: Sequence[Dict[str, Any]],
        temperature: float = 1.0,
        top_k: int = 1,
        save_predictions: bool = True,
        output_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate model on OVO-Bench.
        
        Args:
            samples: List of samples
            temperature: Generation temperature
            top_k: Top-k sampling
            save_predictions: Whether to save predictions
            output_file: Output file for predictions
        
        Returns:
            Results dictionary with metrics
        """
        correct = 0
        total = 0
        predictions = []
        task_results = {}
        
        logger.info(f"Evaluating {len(samples)} samples")
        
        with tqdm(total=len(samples), desc="Evaluating") as pbar:
            for sample in samples:
                # Generate answer via model inference
                frames = sample["frames"]
                answer_text = self._generate_answer(
                    question=sample["question"],
                    options=sample["options"],
                    frames=frames,
                    temperature=temperature,
                    top_k=top_k,
                )
                answer_idx = self._extract_choice(answer_text)

                if answer_idx is None:
                    answer_idx = 0  # default to first option on parse failure
                
                is_correct = answer_idx == sample["answer_idx"]
                correct += int(is_correct)
                total += 1
                
                task_type = sample.get("task_type", "unknown")
                if task_type not in task_results:
                    task_results[task_type] = {"correct": 0, "total": 0}
                task_results[task_type]["correct"] += int(is_correct)
                task_results[task_type]["total"] += 1
                
                # Store prediction
                predictions.append({
                    "video_id": sample["video_id"],
                    "question": sample["question"],
                    "options": sample["options"],
                    "ground_truth": sample["answer_idx"],
                    "predicted": answer_idx,
                    "answer_text": answer_text,
                    "correct": is_correct,
                    "task_type": task_type,
                })
                
                pbar.update(1)
        
        # Compute metrics
        accuracy = correct / total if total > 0 else 0.0
        
        # Per-task accuracy
        per_task_accuracy = {}
        for task_type, results in task_results.items():
            acc = results["correct"] / results["total"] if results["total"] > 0 else 0.0
            per_task_accuracy[task_type] = acc
        
        # Lock vs Fork split
        lock_correct = sum(results["correct"] for task, results in task_results.items() 
                          if task in self.lock_tasks)
        lock_total = sum(results["total"] for task, results in task_results.items() 
                        if task in self.lock_tasks)
        lock_accuracy = lock_correct / lock_total if lock_total > 0 else 0.0
        
        fork_correct = sum(results["correct"] for task, results in task_results.items() 
                          if task in self.fork_tasks)
        fork_total = sum(results["total"] for task, results in task_results.items() 
                        if task in self.fork_tasks)
        fork_accuracy = fork_correct / fork_total if fork_total > 0 else 0.0
        
        results = {
            "overall_accuracy": accuracy,
            "num_correct": correct,
            "num_total": total,
            "per_task_accuracy": per_task_accuracy,
            "lock_accuracy": lock_accuracy,
            "fork_accuracy": fork_accuracy,
            "predictions": predictions if save_predictions else None,
        }
        
        # Save results
        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_file}")
        
        return results


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate SSD-VLM on OVO-Bench")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--data_path", type=str, default="./data/ovo_bench",
                       help="Path to OVO-Bench data")
    parser.add_argument("--output_file", type=str, default="./results/ovo_results.json",
                       help="Output file for results")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load config
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")
    
    # Create evaluator
    evaluator = OVOBenchEvaluator(
        model_path=args.model_path,
        dtype=config["model"].get("dtype", "bfloat16"),
        device_map=config["model"].get("device_map", "auto"),
        num_frames=config["inference"].get("num_frames", 4),
        frame_sampling_strategy=config["inference"].get(
            "frame_sampling_strategy",
            config["evaluation"].get("frame_sampling_strategy", "uniform"),
        ),
        resize_shortest_edge=config["inference"].get(
            "resize_shortest_edge",
            config["evaluation"].get("resize_shortest_edge", 224),
        ),
        max_new_tokens=config["inference"].get("max_new_tokens", 512),
        batch_size=config["data"].get("batch_size", 16),
    )
    
    # Load dataset
    samples = evaluator.load_ovo_dataset(
        data_path=args.data_path,
        split=config["data"].get("split", "test"),
    )
    
    # Evaluate
    results = evaluator.evaluate(
        samples=samples,
        temperature=config["inference"].get("temperature", 1.0),
        top_k=config["inference"].get("top_k", 1),
        save_predictions=config["evaluation"].get("save_predictions", True),
        output_file=args.output_file,
    )
    
    # Print summary
    logger.info(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
    logger.info(f"Lock Task Accuracy: {results['lock_accuracy']:.4f}")
    logger.info(f"Fork Task Accuracy: {results['fork_accuracy']:.4f}")
    
    for task_type, accuracy in results["per_task_accuracy"].items():
        logger.info(f"{task_type} Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
