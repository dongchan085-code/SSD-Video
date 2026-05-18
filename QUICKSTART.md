# SSD-VLM Quick Start Guide

## 5-Minute Setup

### 1. Install
```bash
cd /Users/dongchan_macmini/Documents/Claude/Projects/논문작성
pip install -r requirements.txt
pip install -e .
```

### 2. Download Data (Minimal Test)
```bash
bash scripts/download_data.sh
# Creates placeholder data structure - replace with actual datasets
```

### 2a. OVO-Bench SimpleStream Subsets on Windows

Use this path before SSD training to verify that the SimpleStream baseline code
runs on real OVO-Bench videos without downloading the 142GB pre-chunked archive.
The source-video archive is about 43.16GB, so it fits the 100GB `D:` limit.

```powershell
# Use the Qwen3 official-dependency env. The old C:\Users\...\env_ssd_simplestream
# env is intentionally not used for SimpleStream/Qwen3 reproduction.
$CondaEnv = "D:\conda_envs\env_ssd_simplestream_officialdeps"

# Annotation only; already enough to inspect 1%/10% subset sizes.
conda run -p $CondaEnv python scripts/download_ovo_sources.py `
  --data_root D:/ssd_video_data `
  --skip_parts

# Download the 43.16GB source-video parts once, then prepare the 1% subset.
.\scripts\prepare_ovo_subset_pipeline.ps1 -Ratio 0.01 -DownloadParts -CondaEnv $CondaEnv

# Run the baseline on the 1% subset.
.\scripts\run_simplestream_baseline.ps1 `
  -DataDir D:\ssd_video_data\ovo_subset_1pct `
  -Config .\configs\eval_ovo_subset_1pct_base.yaml `
  -ResultsDir .\results `
  -CondaEnv $CondaEnv

# If the 1% run is healthy, reuse the downloaded parts for the 10% subset.
.\scripts\prepare_ovo_subset_pipeline.ps1 -Ratio 0.10 -CondaEnv $CondaEnv
.\scripts\run_simplestream_baseline.ps1 `
  -DataDir D:\ssd_video_data\ovo_subset_10pct `
  -Config .\configs\eval_ovo_subset_10pct_base.yaml `
  -ResultsDir .\results `
  -CondaEnv $CondaEnv
```

### 3. Generate Figures (with mock data)
```bash
python figures/plot_all.py --use_mock_data --output_dir ./figures/outputs
# Creates 3 publication-quality figures instantly
```

## Full Pipeline (8 GPUs)

```bash
bash scripts/run_full_pipeline.sh ./data ./outputs ./results 8
```

This runs:
1. Data prep
2. SSD sampling (frozen model, high-temp)
3. LoRA fine-tuning (paper method)
4. OVO-Bench evaluation
6. Ablation studies (frame, temperature sweeps)
7. Figure generation
8. Results aggregation

Estimated time: 48-72 hours on 8x A100

## Individual Commands

### SSD Sampling
```bash
python ssd_vlm/sampling/generate_samples.py \
  --config configs/sample_generation.yaml \
  --output_dir ./outputs/ssd_samples \
  --data_path ./data/perception_test
```

### LoRA Training (8 GPUs)
```bash
torchrun --nproc_per_node=8 \
  ssd_vlm/training/train_lora.py \
  --config configs/train_lora.yaml \
  --samples_path ./outputs/ssd_samples/samples.jsonl \
  --output_dir ./outputs/lora_checkpoint
```

### Evaluate Base Model
```bash
python eval/eval_ovo_bench.py \
  --config configs/eval_ovo_base.yaml \
  --model_path "Qwen/Qwen3-VL-8B-Instruct" \
  --data_path ./data/ovo_bench \
  --output_file ./results/ovo_base.json
```

### Evaluate SSD-VLM
```bash
python eval/eval_ovo_bench.py \
  --config configs/eval_ovo_ssd.yaml \
  --model_path ./outputs/ssd_vlm_final \
  --data_path ./data/ovo_bench \
  --output_file ./results/ovo_ssd.json
```

### Frame Budget Sweep
```bash
python eval/eval_frame_sweep.py \
  --config configs/eval_ovo_frame_sweep.yaml \
  --model_path ./outputs/ssd_vlm_final \
  --data_path ./data/ovo_bench \
  --output_dir ./results/frame_sweep
```

### Temperature Sweep
```bash
python eval/eval_temperature_sweep.py \
  --config configs/eval_temperature_sweep.yaml \
  --model_path "Qwen/Qwen3-VL-8B-Instruct" \
  --data_path ./data/ovo_bench \
  --output_dir ./results/temperature_sweep
```

### Score Results
```bash
python eval/score_results.py \
  --base_results ./results/ovo_base.json \
  --ssd_results ./results/ovo_ssd.json \
  --frame_sweep_dir ./results/frame_sweep \
  --temperature_sweep_dir ./results/temperature_sweep \
  --output_file ./results/scored_results.json
```

## Key Files to Modify

### Hyperparameters
- `configs/sample_generation.yaml` - Sampling params (temperature, top-k)
- `configs/train_lora.yaml` - LoRA rank, learning rate, epochs
- `configs/train_full_ft.yaml` - FT learning rate, batch size (ablation only)

### Model Paths
All configs use relative paths. Update if using different directory structure:
- `data_path`: Dataset location
- `output_dir`: Checkpoint location
- `model_path`: Pretrained model

### GPU Configuration
Edit script or pass arguments:
```bash
bash scripts/run_full_pipeline.sh ./data ./outputs ./results NUM_GPUS
```

## Expected Output Structure

```
outputs/
├── ssd_samples/
│   └── samples.jsonl              # Raw model completions
├── lora_checkpoint/
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── ...
results/
├── ovo_base.json                  # Base model results
├── ovo_ssd.json                   # SSD-VLM results
├── scored_results.json            # Aggregated metrics
├── frame_sweep/
│   └── frame_sweep_results.json
└── temperature_sweep/
    └── temperature_sweep_results.json

figures/outputs/
├── perception_memory_tradeoff.pdf
├── lock_fork_asymmetry.pdf
└── temperature_sweep.pdf
```

## Troubleshooting

### Out of Memory
1. Reduce `per_device_train_batch_size` in config
2. Increase `gradient_accumulation_steps`
3. Enable gradient checkpointing
4. Use DeepSpeed ZeRO-3 (see ablation configs)

### Slow Data Loading
1. Check `num_workers` in data config
2. Pre-extract frames to local SSD storage
3. Enable frame caching (default: enabled)

### Training Instability
1. Reduce learning rate by 2x
2. Increase warmup steps
3. Check sample quality from generation phase

### Missing Dependencies
```bash
pip install --upgrade transformers torch deepspeed
```

## Monitoring Training

Check TensorBoard logs:
```bash
tensorboard --logdir=./outputs/lora_checkpoint/runs
```

Or WandB logs (if enabled):
- Set `wandb.project` in config
- Log in: `wandb login`

## Configuration Explanation

### sample_generation.yaml
```yaml
generation:
  temperature: 1.5      # High temperature for diversity
  top_k: 10            # Restrict to top 10 tokens
  do_sample: true      # Enable sampling

data:
  num_frames: 4        # Frames per video (fixed)
  memory_skill_oversample_ratio: 2.0  # 2x oversampling
```

### train_lora.yaml
```yaml
lora:
  r: 128               # Rank (model capacity)
  lora_alpha: 256      # Scaling factor (usually 2x rank)
  
training:
  num_train_epochs: 2
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8  # Effective batch: 16 per GPU
  learning_rate: 5e-4
```

## Expected Results

SSD-VLM targets the Pareto expansion region (ΔRT ≥ 0, ΔMem > 0):
- **Fork (memory) tasks**: primary improvement target
- **Lock (perception) tasks**: accuracy preserved or marginally improved
- **Zero additional latency**: merged LoRA adds no inference overhead

## Paper Figures

The visualization scripts create:
1. **Perception-Memory Tradeoff**: Green zone showing SSD-VLM sweet spot
2. **Lock vs Fork Asymmetry**: Bar chart of per-task improvements
3. **Temperature Plateau**: Line plot showing SSD stability vs base plateau

All figures use publication-quality styling (Nature journal style).

## Citation

```bibtex
@article{ssd-vlm-2025,
  title={Simple Self-Distillation for Efficient Vision Language Models},
  author={Your Name},
  year={2025}
}
```

## Support

For issues or questions:
1. Check README.md for detailed documentation
2. Review PROJECT_SUMMARY.md for architecture
3. Check specific config files for parameter explanations
4. Enable debug logging: set `log_level: DEBUG` in configs

---

**Last Updated**: April 2026
**Framework**: PyTorch 2.1+ / Transformers 4.40+
**Hardware**: A100/H100 (adjust batch_size for smaller GPUs)
