"""
Utilities for loading base and LoRA-adapted VLM checkpoints.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

try:
    from transformers import BitsAndBytesConfig
except ImportError:  # pragma: no cover - depends on transformers version
    BitsAndBytesConfig = None


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


def normalize_device_map(device_map: Union[str, Dict]) -> Union[str, Dict]:
    """Allow configs to request strict single-GPU placement with device_map: cuda."""
    if isinstance(device_map, str) and device_map.lower() in {"cuda", "cuda:0", "gpu"}:
        return {"": 0}
    return device_map


def load_vlm_processor_and_model(
    model_path: str,
    dtype: str = "bfloat16",
    device_map: str = "auto",
    max_memory: Optional[Dict[Union[int, str], str]] = None,
    load_in_8bit: bool = False,
    trust_remote_code: bool = True,
    merge_lora: bool = True,
) -> Tuple[AutoProcessor, AutoModelForImageTextToText]:
    """
    Load a processor/model pair from either a base checkpoint or a LoRA adapter.
    """
    torch_dtype = resolve_torch_dtype(dtype)
    device_map = normalize_device_map(device_map)
    quantization_config = None
    if load_in_8bit:
        if BitsAndBytesConfig is None:
            raise ImportError("transformers BitsAndBytesConfig is unavailable; cannot load in 8bit.")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )

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
            max_memory=max_memory,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
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
            max_memory=max_memory,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            trust_remote_code=trust_remote_code,
        )

    return processor, model
