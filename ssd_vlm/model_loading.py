"""
Utilities for loading base and LoRA-adapted VLM checkpoints.
"""

from pathlib import Path
from typing import Tuple

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor


def resolve_torch_dtype(dtype: str) -> torch.dtype:
    """Resolve a string dtype to a torch dtype."""
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return dtype_map.get(dtype, torch.float32)


def is_peft_adapter_path(model_path: str) -> bool:
    """Detect whether a checkpoint path is a PEFT adapter directory."""
    path = Path(model_path)
    return path.is_dir() and (path / "adapter_config.json").exists()


def load_vlm_processor_and_model(
    model_path: str,
    dtype: str = "bfloat16",
    device_map: str = "auto",
    trust_remote_code: bool = True,
    merge_lora: bool = True,
) -> Tuple[AutoProcessor, AutoModelForImageTextToText]:
    """
    Load a processor/model pair from either a base checkpoint or a LoRA adapter.
    """
    torch_dtype = resolve_torch_dtype(dtype)

    if is_peft_adapter_path(model_path):
        peft_config = PeftConfig.from_pretrained(model_path)
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )
        base_model = AutoModelForImageTextToText.from_pretrained(
            peft_config.base_model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        if merge_lora:
            model = model.merge_and_unload()
    else:
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )

    return processor, model
