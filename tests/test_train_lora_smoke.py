"""LoRA training pipeline smoke test (R12).

Instantiates LoRATrainer with the mini CPU config, runs exactly 1 training
step on a synthetic text-only batch, and asserts the loss is finite.
No GPU, no OVO-Bench data, no Perception Test data required.

Run:
    pytest tests/test_train_lora_smoke.py -v -m slow
    # or without -m to include in the full suite (will be slow on first run
    # due to Qwen3-VL-2B model download ~5 GB)

Mark: slow  (skipped in fast CI, run manually or on the 3050 machine)
"""

import math
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

transformers = pytest.importorskip("transformers", reason="transformers not installed")
pytest.importorskip("peft", reason="peft not installed")

from ssd_vlm.training.train_lora import LoRATrainer
from ssd_vlm.utils.config import load_config


class _SyntheticTextLoader:
    """One-shot DataLoader that yields a single text-only batch."""
    def __init__(self, seq_len: int = 16, vocab_size: int = 32000):
        tok = torch.randint(1, vocab_size, (1, seq_len))
        self._batch = {
            "input_ids": tok,
            "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
            "labels": tok.clone(),
        }
    def __iter__(self):
        yield self._batch
    def __len__(self):
        return 1


@pytest.mark.slow
class TestLoRATrainerSmoke:
    def test_one_step_cpu(self, tmp_path):
        cfg_path = PROJECT_ROOT / "configs" / "train_lora_mini_3050.yaml"
        if not cfg_path.exists():
            pytest.skip(f"Config not found: {cfg_path}")

        config = load_config(str(cfg_path))
        model_id = config["model"].get("model_id", "Qwen/Qwen3-VL-2B-Instruct")

        try:
            trainer = LoRATrainer(
                model_id=model_id,
                output_dir=str(tmp_path / "lora_smoke"),
                lora_config=config["lora"],
                training_config=config["training"],
                model_config=config.get("model"),
                device="cpu",
            )
        except Exception as exc:
            msg = str(exc).lower()
            if any(k in msg for k in ("not found", "404", "no such file", "connection")):
                pytest.skip(f"Model weights unavailable: {exc}")
            raise

        loader = _SyntheticTextLoader()
        trainer.train(loader)

        # Verify loss was computed and is finite
        assert hasattr(trainer, "last_loss"), "LoRATrainer must expose last_loss after train()"
        assert math.isfinite(trainer.last_loss), f"loss is not finite: {trainer.last_loss}"
