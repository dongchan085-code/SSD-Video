"""GPU inference smoke for RTX 3050 4GB — Qwen3-VL-2B-Instruct + NF4 + SDPA.

Run:
    python tests/smoke_gpu_3050.py

Verifies:
  1. NF4 model loads within the 4GB VRAM budget.
  2. All parameters are on CUDA (no silent CPU spill).
  3. Text-only generation works.
  4. Video-frame generation (4 synthetic frames) works.

Exit 0 on success; exit 1 on OOM or assertion failure.
"""

import argparse
import gc
import sys
import time

import numpy as np
import torch
from PIL import Image

_REPO = __file__.rsplit("tests", 1)[0]
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from ssd_vlm.model_loading import load_vlm_processor_and_model
from ssd_vlm.utils.config import load_config


def gb(n: int) -> float:
    return n / 1024 ** 3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model_id", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--config", default="configs/_3050_4gb.yaml",
                        help="Profile YAML (model.* keys used for loading)")
    parser.add_argument("--budget_gb", type=float, default=3.5,
                        help="Abort if peak VRAM exceeds this (default 3.5 for 4GB card)")
    parser.add_argument("--gen_tokens", type=int, default=16)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("[SKIP] No CUDA device — run on the 3050 machine.")

    cfg = load_config(args.config).get("model", {})
    torch.cuda.reset_peak_memory_stats()
    total_gb = gb(torch.cuda.mem_get_info()[1])
    print(f"[start] GPU total={total_gb:.2f}GB  budget={args.budget_gb:.1f}GB")

    t0 = time.time()
    processor, model = load_vlm_processor_and_model(
        model_path=args.model_id,
        dtype=cfg.get("dtype", "float16"),
        device_map=cfg.get("device_map", "cuda"),
        load_in_4bit=cfg.get("load_in_4bit", True),
        load_in_8bit=cfg.get("load_in_8bit", False),
        attn_implementation=cfg.get("attn_implementation", "sdpa"),
        max_pixels=cfg.get("max_pixels"),
        min_pixels=cfg.get("min_pixels"),
    )
    model.eval()
    load_s = time.time() - t0
    peak_load = gb(torch.cuda.max_memory_allocated())
    print(f"[load] {load_s:.1f}s  peak={peak_load:.2f}GB")

    cpu_params = [n for n, p in model.named_parameters() if p.device.type != "cuda"]
    if cpu_params:
        sys.exit(f"[FAIL] {len(cpu_params)} params on CPU: {cpu_params[:3]}")
    print("[ok] all params on cuda")

    if peak_load > args.budget_gb:
        sys.exit(f"[FAIL] load peak {peak_load:.2f}GB > budget {args.budget_gb:.2f}GB")

    # Text-only smoke
    msgs = [{"role": "user", "content": [{"type": "text", "text": "Say 'hi' and stop."}]}]
    inputs = processor.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )
    inputs.pop("token_type_ids", None)
    inputs = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=args.gen_tokens, do_sample=False, use_cache=True)
    text = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"[text] peak={gb(torch.cuda.max_memory_allocated()):.2f}GB  out={text!r:.40s}")

    # Video-frame smoke — 4 synthetic frames
    frames = [Image.fromarray(np.random.randint(0, 255, (224, 398, 3), dtype=np.uint8)) for _ in range(4)]
    msgs = [{"role": "user", "content": [
        {"type": "video", "video": frames},
        {"type": "text", "text": "What's in the clip? Answer A, B, C, or D."},
    ]}]
    inputs = processor.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )
    inputs.pop("token_type_ids", None)
    inputs = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    prompt_len = int(inputs["input_ids"].shape[1])
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=args.gen_tokens, do_sample=False, use_cache=True)
    peak_vid = gb(torch.cuda.max_memory_allocated())
    text = processor.decode(out[0][prompt_len:], skip_special_tokens=True)
    print(f"[video] peak={peak_vid:.2f}GB  prompt_tok={prompt_len}  out={text!r:.40s}")

    if peak_vid > args.budget_gb:
        sys.exit(f"[FAIL] video peak {peak_vid:.2f}GB > budget {args.budget_gb:.2f}GB")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("[PASS] all checks passed")


if __name__ == "__main__":
    main()
