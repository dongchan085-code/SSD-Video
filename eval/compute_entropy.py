"""
Compute per-token output entropy and answer-token rank for Lock-Fork
hypothesis validation (Section 4 of the paper).

Usage:
    python eval/compute_entropy.py \
        --model_path Qwen/Qwen3-VL-8B-Instruct \
        --data_dir data/ovo_bench \
        --output results/entropy_base.json \
        --num_frames 4
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

logger = logging.getLogger(__name__)

# ── Task taxonomy (shared across evaluation scripts) ────────────────
LOCK_TASKS = {"OCR", "ATR", "OJR", "STU", "ACR", "FPD"}
FORK_TASKS = {"EPM", "ASI", "HLD"}

ANSWER_TOKENS = ["A", "B", "C", "D"]


# ── Dataset loading (reused from eval_ovo_bench.py) ─────────────────

def load_ovo_dataset(data_path: str, split: str = "test") -> List[Dict[str, Any]]:
    """Load OVO-Bench annotations."""
    split_file = Path(data_path) / f"{split}_split.json"
    annotations_file = Path(data_path) / f"{split}_annotations.json"

    if not split_file.exists() or not annotations_file.exists():
        raise FileNotFoundError(f"OVO-Bench data not found in {data_path}")

    with open(split_file) as f:
        split_data = json.load(f)
    with open(annotations_file) as f:
        annotations = json.load(f)

    samples = []
    for video_id in split_data.get("video_ids", []):
        if video_id not in annotations:
            continue
        ann = annotations[video_id]
        samples.append({
            "video_id": video_id,
            "question": ann.get("question", ""),
            "options": ann.get("options", []),
            "answer_idx": ann.get("answer_idx", 0),
            "task_type": ann.get("task_type", ""),
        })
    return samples


def format_prompt(question: str, options: List[str]) -> str:
    """Format question + options into a QA prompt."""
    options_text = "\n".join(
        f"{chr(65 + i)}: {opt}" for i, opt in enumerate(options)
    )
    return f"Question: {question}\n\nOptions:\n{options_text}\n\nAnswer:"


# ── Core entropy computation ────────────────────────────────────────

class EntropyComputer:
    """Compute per-token entropy and answer-token rank via forward pass."""

    def __init__(
        self,
        model_path: str,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        num_frames: int = 4,
    ):
        self.num_frames = num_frames

        torch_dtype = {"bfloat16": torch.bfloat16,
                       "float16": torch.float16}.get(dtype, torch.float32)

        logger.info(f"Loading model: {model_path}")
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.model.eval()

        # Cache answer-token IDs (A / B / C / D)
        self.answer_token_ids = []
        for tok in ANSWER_TOKENS:
            ids = self.processor.tokenizer.encode(tok, add_special_tokens=False)
            self.answer_token_ids.append(ids[-1])   # last sub-token
        logger.info(f"Answer token IDs: {dict(zip(ANSWER_TOKENS, self.answer_token_ids))}")

    @torch.no_grad()
    def _forward_sample(
        self,
        question: str,
        options: List[str],
        frames: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Run a single forward pass and return entropy + rank at the
        answer position (last token of the prompt).

        Returns dict with keys: entropy, rank, gt_prob
        """
        prompt = format_prompt(question, options)

        messages = [{"role": "user", "content": [
            {"type": "text", "text": prompt},
        ]}]

        # If frames are available, prepend image content
        if frames is not None:
            messages[0]["content"].insert(
                0, {"type": "image", "image": frames})

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

        inputs = self.processor(
            text=[text],
            images=[frames] if frames is not None else None,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.model.device) if torch.is_tensor(v) else v
                  for k, v in inputs.items()}

        outputs = self.model(**inputs)
        # logits at the last prompt token → distribution over next token
        last_logits = outputs.logits[0, -1, :]  # [vocab_size]

        # Shannon entropy
        probs = F.softmax(last_logits.float(), dim=-1)
        probs_clamped = probs.clamp(min=1e-12)
        entropy = -torch.sum(probs_clamped * torch.log(probs_clamped)).item()

        return {"entropy": entropy, "probs": probs}

    def _rank_of_gt(self, probs: torch.Tensor, answer_idx: int) -> int:
        """Return 1-indexed rank of the ground-truth answer token."""
        gt_token_id = self.answer_token_ids[answer_idx]
        sorted_ids = torch.argsort(probs, descending=True)
        rank = (sorted_ids == gt_token_id).nonzero(as_tuple=True)[0]
        return (rank.item() + 1) if rank.numel() > 0 else -1

    def compute(
        self,
        samples: List[Dict[str, Any]],
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run entropy analysis over all samples."""
        lock_entropies: List[float] = []
        fork_entropies: List[float] = []
        lock_ranks: List[int] = []
        fork_ranks: List[int] = []
        per_sample: List[Dict[str, Any]] = []

        for sample in tqdm(samples, desc="Computing entropy"):
            result = self._forward_sample(
                sample["question"], sample["options"])
            entropy = result["entropy"]
            rank = self._rank_of_gt(result["probs"], sample["answer_idx"])

            task = sample["task_type"]
            if task in LOCK_TASKS:
                lock_entropies.append(entropy)
                lock_ranks.append(rank)
            elif task in FORK_TASKS:
                fork_entropies.append(entropy)
                fork_ranks.append(rank)

            per_sample.append({
                "video_id": sample["video_id"],
                "task_type": task,
                "entropy": entropy,
                "rank": rank,
            })

        def _stats(vals):
            a = np.array(vals, dtype=float)
            return {"mean": float(a.mean()), "std": float(a.std()),
                    "count": len(a), "values": a.tolist()} if len(a) else {}

        out = {
            "lock_entropy": _stats(lock_entropies),
            "fork_entropy": _stats(fork_entropies),
            "lock_rank": _stats(lock_ranks),
            "fork_rank": _stats(fork_ranks),
            "per_sample": per_sample,
        }

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(out, f, indent=2)
            logger.info(f"Saved to {output_path}")

        return out


# ── CLI ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute per-token entropy for Lock-Fork hypothesis")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_dir", default="./data/ovo_bench")
    parser.add_argument("--output", default="./results/entropy.json")
    parser.add_argument("--num_frames", type=int, default=4)
    parser.add_argument("--split", default="test")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    samples = load_ovo_dataset(args.data_dir, args.split)
    logger.info(f"Loaded {len(samples)} samples")

    computer = EntropyComputer(
        model_path=args.model_path,
        dtype=args.dtype,
        num_frames=args.num_frames,
    )
    results = computer.compute(samples, output_path=args.output)

    logger.info("=== Entropy Summary ===")
    for cat in ("lock", "fork"):
        e = results[f"{cat}_entropy"]
        r = results[f"{cat}_rank"]
        if e:
            logger.info(f"  {cat.upper()} entropy: {e['mean']:.4f} +/- {e['std']:.4f}  "
                        f"rank: {r['mean']:.1f} +/- {r['std']:.1f}  (n={e['count']})")


if __name__ == "__main__":
    main()
