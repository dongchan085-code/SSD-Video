import json

with open("results/ovo_1pct/ovo_base_qwen2vl2b_gpu_1pct.json") as f:
    prev = json.load(f)
with open("results/ovo_1pct/qwen3vl8b_nf4_t4.json") as f:
    cur = json.load(f)

print(f"{'metric':<25} {'Qwen2VL-2B':<14} {'Qwen3VL-8B NF4':<14}")
print("-" * 53)
for k in [
    "overall_accuracy",
    "lock_accuracy",
    "fork_accuracy",
    "realtime_accuracy",
    "backward_accuracy",
    "forward_accuracy",
    "rt_bwd_avg",
    "ovo_total_avg_3way",
    "mean_latency_ms",
    "peak_gpu_memory_gb",
    "throughput_samples_per_sec",
]:
    a = prev.get(k)
    b = cur.get(k)
    sa = f"{a:.4f}" if isinstance(a, float) else str(a)
    sb = f"{b:.4f}" if isinstance(b, float) else str(b)
    print(f"{k:<25} {sa:<14} {sb:<14}")

print()
print("per-task accuracy:")
print(f"{'task':<8} {'Qwen2VL-2B':<14} {'Qwen3VL-8B NF4':<14}")
keys = sorted(set(list(prev["per_task_accuracy"].keys()) + list(cur["per_task_accuracy"].keys())))
for t in keys:
    pa = prev["per_task_accuracy"].get(t)
    pb = cur["per_task_accuracy"].get(t)
    sa = f"{pa:.3f}" if pa is not None else "-"
    sb = f"{pb:.3f}" if pb is not None else "-"
    print(f"{t:<8} {sa:<14} {sb:<14}")
