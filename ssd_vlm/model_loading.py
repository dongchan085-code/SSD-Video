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


def _build_quantization_config(load_in_8bit: bool, load_in_4bit: bool, compute_dtype: torch.dtype):
    if not (load_in_8bit or load_in_4bit):
        return None
    if BitsAndBytesConfig is None:
        raise ImportError("transformers BitsAndBytesConfig is unavailable; install bitsandbytes.")
    if load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["visual", "lm_head"],
        )
    return BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_skip_modules=["visual", "lm_head"],
    )


def _apply_pixel_caps(processor, max_pixels: Optional[int], min_pixels: Optional[int]) -> None:
    """Clamp the processor's image/video token budget — set after load to avoid silently being ignored."""
    if max_pixels is None and min_pixels is None:
        return
    size: Dict[str, int] = {}
    if max_pixels is not None:
        size["longest_edge"] = int(max_pixels)
    if min_pixels is not None:
        size["shortest_edge"] = int(min_pixels)
    for attr in ("image_processor", "video_processor"):
        sub = getattr(processor, attr, None)
        if sub is None:
            continue
        try:
            sub.size = dict(size)
        except Exception:
            pass
        for cap_name, cap_value in (("max_pixels", max_pixels), ("min_pixels", min_pixels)):
            if cap_value is not None and hasattr(sub, cap_name):
                try:
                    setattr(sub, cap_name, int(cap_value))
                except Exception:
                    pass


def load_vlm_processor_and_model(
    model_path: str,
    dtype: str = "bfloat16",
    device_map: str = "auto",
    max_memory: Optional[Dict[Union[int, str], str]] = None,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    attn_implementation: Optional[str] = None,
    max_pixels: Optional[int] = None,
    min_pixels: Optional[int] = None,
    trust_remote_code: bool = True,
    merge_lora: bool = True,
) -> Tuple[AutoProcessor, AutoModelForImageTextToText]:
    """
    Load a processor/model pair from either a base checkpoint or a LoRA adapter.
    """
    torch_dtype = resolve_torch_dtype(dtype)
    device_map = normalize_device_map(device_map)
    quantization_config = _build_quantization_config(load_in_8bit, load_in_4bit, torch_dtype)

    common_kwargs: Dict[str, object] = dict(
        torch_dtype=torch_dtype,
        device_map=device_map,
        max_memory=max_memory,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    )
    if attn_implementation:
        common_kwargs["attn_implementation"] = attn_implementation

    if is_peft_adapter_path(model_path):
        peft_config = PeftConfig.from_pretrained(model_path)
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )
        base_model = AutoModelForImageTextToText.from_pretrained(
            peft_config.base_model_name_or_path,
            **common_kwargs,
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        if merge_lora and quantization_config is None:
            model = model.merge_and_unload()
    else:
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            **common_kwargs,
        )

    _apply_pixel_caps(processor, max_pixels, min_pixels)
    return processor, model
