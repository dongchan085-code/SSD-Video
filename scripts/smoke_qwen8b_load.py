"""Smoke-test Qwen3-VL-8B loading before running OVO evaluation."""

import argparse
import gc
import json
import os
from pathlib import Path

import torch
import yaml

from ssd_vlm.model_loading import load_vlm_processor_and_model


def cuda_snapshot(label: str) -> None:
    if not torch.cuda.is_available():
        print(f"[{label}] cuda unavailable", flush=True)
        return
    free, total = torch.cuda.mem_get_info()
    print(
        f"[{label}] cuda free={free / (1024 ** 3):.2f}GiB "
        f"total={total / (1024 ** 3):.2f}GiB "
        f"allocated={torch.cuda.memory_allocated() / (1024 ** 3):.2f}GiB "
        f"reserved={torch.cuda.memory_reserved() / (1024 ** 3):.2f}GiB",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Load the configured VLM and print memory/device-map info.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_path", default=None)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model_cfg = config["model"]
    model_path = args.model_path or model_cfg["model_id"]

    print("[env] HF_HOME=", os.environ.get("HF_HOME"), flush=True)
    print("[env] PYTORCH_CUDA_ALLOC_CONF=", os.environ.get("PYTORCH_CUDA_ALLOC_CONF"), flush=True)
    print("[config]", json.dumps(model_cfg, indent=2), flush=True)
    cuda_snapshot("before_load")

    processor, model = load_vlm_processor_and_model(
        model_path=model_path,
        dtype=model_cfg.get("dtype", "float16"),
        device_map=model_cfg.get("device_map", "auto"),
        max_memory=model_cfg.get("max_memory"),
        load_in_8bit=model_cfg.get("load_in_8bit", False),
    )
    model.eval()
    cuda_snapshot("after_load")
    print("[device_map]", getattr(model, "hf_device_map", None), flush=True)
    print("[processor]", type(processor).__name__, flush=True)

    del model
    del processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    cuda_snapshot("after_cleanup")


if __name__ == "__main__":
    main()
