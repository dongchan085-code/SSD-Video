"""Smoke test for T4 OOM fix: load Qwen3-VL-8B with NF4 + sdpa and report peak VRAM.

Run:  python tests/smoke_oom.py
"""
import argparse
import gc
import time

import torch

from ssd_vlm.model_loading import load_vlm_processor_and_model


def gb(x: int) -> float:
    return x / 1024**3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device_map", default="cuda")
    parser.add_argument("--quant", default="nf4", choices=["nf4", "int8", "none"])
    parser.add_argument("--attn", default="sdpa")
    parser.add_argument("--max_pixels", type=int, default=200704)
    parser.add_argument("--min_pixels", type=int, default=3136)
    parser.add_argument("--gen_tokens", type=int, default=16)
    parser.add_argument("--budget_gb", type=float, default=14.5, help="abort if peak exceeds")
    args = parser.parse_args()

    torch.cuda.reset_peak_memory_stats()
    print(f"[start] free={gb(torch.cuda.mem_get_info()[0]):.2f}GB  total={gb(torch.cuda.mem_get_info()[1]):.2f}GB")

    load_in_4bit = args.quant == "nf4"
    load_in_8bit = args.quant == "int8"

    t0 = time.time()
    processor, model = load_vlm_processor_and_model(
        model_path=args.model_id,
        dtype=args.dtype,
        device_map=args.device_map,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        attn_implementation=args.attn,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
    )
    model.eval()
    load_s = time.time() - t0
    print(f"[load done] {load_s:.1f}s  allocated={gb(torch.cuda.memory_allocated()):.2f}GB  peak={gb(torch.cuda.max_memory_allocated()):.2f}GB")

    # Verify every param is on cuda — catch silent CPU spill.
    cpu_params = [n for n, p in model.named_parameters() if p.device.type != "cuda"]
    if cpu_params:
        print(f"[FAIL] {len(cpu_params)} parameters on CPU. First: {cpu_params[:5]}")
        return
    print("[ok] all parameters on cuda")

    peak_after_load = gb(torch.cuda.max_memory_allocated())
    if peak_after_load > args.budget_gb:
        print(f"[ABORT] peak {peak_after_load:.2f}GB exceeds budget {args.budget_gb:.2f}GB before inference")
        return

    # Text-only generate smoke
    msgs = [{"role": "user", "content": [{"type": "text", "text": "Say 'hi' and stop."}]}]
    inputs = processor.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )
    inputs.pop("token_type_ids", None)
    inputs = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=args.gen_tokens, do_sample=False, use_cache=True)
    gen_s = time.time() - t0
    text = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    text_safe = text.encode("ascii", "replace").decode("ascii")
    print(f"[text gen] {gen_s:.1f}s  peak={gb(torch.cuda.max_memory_allocated()):.2f}GB  out={text_safe!r}")

    # Video-frames inference smoke — 4 synthetic frames, mirrors the actual pipeline shape
    from PIL import Image
    import numpy as np
    frames = [Image.fromarray(np.random.randint(0, 255, (224, 398, 3), dtype=np.uint8)) for _ in range(4)]
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frames},
                {"type": "text", "text": "What's in this clip? Answer with one letter A, B, C, or D."},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )
    inputs.pop("token_type_ids", None)
    inputs = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    prompt_tokens = int(inputs["input_ids"].shape[1])
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=args.gen_tokens, do_sample=False, use_cache=True)
    gen_s = time.time() - t0
    text = processor.decode(out[0][prompt_tokens:], skip_special_tokens=True)
    text_safe = text.encode("ascii", "replace").decode("ascii")
    print(
        f"[video gen] {gen_s:.1f}s  prompt_tok={prompt_tokens}  "
        f"peak={gb(torch.cuda.max_memory_allocated()):.2f}GB  out={text_safe!r}"
    )

    if gb(torch.cuda.max_memory_allocated()) > args.budget_gb:
        print(f"[WARN] peak exceeded budget {args.budget_gb:.2f}GB")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("[end]")


if __name__ == "__main__":
    main()
