# DPFM Ablation Experiments - Final Documentation

## Overview

This document tracks all experiments comparing **Diffusion Policy (DDPM)** and **Flow Matching (FM)** on the PushT benchmark task.

## WandB Projects

All experiments log to WandB with distinct run names:

| Project | URL | Description |
|---------|-----|-------------|
| `dpfm_pusht_v2` | https://wandb.ai/danny010324/dpfm_pusht_v2 | FM experiments (currently running) |
| `dpfm_pusht_experiments` | https://wandb.ai/danny010324/dpfm_pusht_experiments | DDPM experiments (currently running) |
| `dpfm_pusht_ablation` | (future runs) | Unified project for all future experiments |

### Current Run Names in WandB

| Job ID | WandB Run Name | WandB Project | Description |
|--------|----------------|---------------|-------------|
| 2526314 | `ddpm_unet_seed42` | dpfm_pusht_experiments | DDPM UNet baseline |
| 2525656 | `fm_unet_step4_seed42` | dpfm_pusht_v2 | FM UNet baseline |
| 2525657 | `fm_transformer_transformer_step4_seed42` | dpfm_pusht_v2 | FM Transformer baseline |
| 2525658 | `fm_steps4_step4_seed42` | dpfm_pusht_v2 | FM 4 inference steps |
| 2525659 | `fm_steps8_step8_seed42` | dpfm_pusht_v2 | FM 8 inference steps |
| 2525660 | `fm_steps16_step16_seed42` | dpfm_pusht_v2 | FM 16 inference steps |

---

## Currently Running Experiments

### Phase 1: Baselines + Ablation A (Started 2025-12-04)

| Job ID | SLURM Name | Method | Architecture | Inference Steps | Node | Status |
|--------|------------|--------|--------------|-----------------|------|--------|
| 2526314 | ddpm_unet_s42 | DDPM | UNet | 100 | atl1-1-03-004-29-0 | ✅ Training |
| 2525656 | fm_unet_s42 | FM | UNet | 4 | atl1-1-03-007-29-0 | ✅ Training |
| 2525657 | fm_trans_s42 | FM | Transformer | 4 | atl1-1-03-007-31-0 | ✅ Training |
| 2525658 | fm_steps4 | FM | UNet | 4 | atl1-1-01-010-29-0 | ✅ Training |
| 2525659 | fm_steps8 | FM | UNet | 8 | atl1-1-01-010-31-0 | ✅ Training |
| 2525660 | fm_steps16 | FM | UNet | 16 | atl1-1-01-010-33-0 | ✅ Training |

**Total: 6 jobs running**

### Output Directories

Each job creates outputs in `data/outputs/YYYY.MM.DD/HH.MM.SS_<config_name>_pusht_image/`:

| Job | Output Directory Pattern |
|-----|--------------------------|
| ddpm_unet_s42 | `data/outputs/2025.12.04/*_train_ddpm_unet_hybrid_pusht_image/` |
| fm_unet_s42 | `data/outputs/2025.12.04/*_train_fm_unet_hybrid_pusht_image/` |
| fm_trans_s42 | `data/outputs/2025.12.04/*_train_fm_transformer_hybrid_pusht_image/` |

---

## Experiment Design

### Baselines (3 experiments)
| Experiment | Method | Architecture | Inference Steps | Epochs |
|------------|--------|--------------|-----------------|--------|
| `ddpm_unet_s42` | DDPM | UNet | 100 | 3050 |
| `fm_unet_s42` | FM | UNet | 4 | 3050 |
| `fm_trans_s42` | FM | Transformer | 4 | 3050 |

### Ablation A: Inference Steps (3 experiments)
| Experiment | Inference Steps |
|------------|-----------------|
| `fm_steps4` | 4 steps (baseline) |
| `fm_steps8` | 8 steps |
| `fm_steps16` | 16 steps |

### Ablation B: Learning Rate + Warmup (4 experiments - pending)
| Experiment | Learning Rate | Warmup Steps |
|------------|---------------|--------------|
| `fm_lr1e-4_w500` | 1e-4 | 500 (baseline) |
| `fm_lr1e-4_w1000` | 1e-4 | 1000 |
| `fm_lr2e-4_w500` | 2e-4 | 500 |
| `fm_lr2e-4_w1000` | 2e-4 | 1000 |

### Ablation C: Seed Sensitivity (5 experiments - pending)
| Experiment | Seed |
|------------|------|
| `fm_seed42` | 42 |
| `fm_seed43` | 43 |
| `fm_seed44` | 44 |
| `fm_seed45` | 45 |
| `fm_seed46` | 46 |

### Ablation D: Dataset Size (6 experiments - pending)
| Experiment | Method | Episodes |
|------------|--------|----------|
| `fm_data90` | FM | 90 (full) |
| `fm_data60` | FM | 60 |
| `fm_data30` | FM | 30 |
| `ddpm_data90` | DDPM | 90 (full) |
| `ddpm_data60` | DDPM | 60 |
| `ddpm_data30` | DDPM | 30 |

---

## Previous Training Results

### Best FM Models (from earlier experiments)

| Path | Best Score | Epoch | Notes |
|------|------------|-------|-------|
| `data/outputs/2025.11.27/04.36.04_train_fm_unet_hybrid_pusht_image/` | **0.845** | 900 | FM UNet, best overall |
| `data/outputs/2025.11.27/14.28.45_train_fm_unet_hybrid_pusht_image/` | 0.838 | 200 | FM UNet |
| `data/outputs/2025.11.29/00.32.47_train_fm_unet_hybrid_pusht_image/` | 0.823 | 650 | FM UNet |

### Best Checkpoint for Evaluation
```
data/outputs/2025.11.27/04.36.04_train_fm_unet_hybrid_pusht_image/checkpoints/epoch=0900-test_mean_score=0.845.ckpt
```

---

## Cluster Configuration

| Setting | Value |
|---------|-------|
| Cluster | Georgia Tech PACE Phoenix |
| Account | gts-agarg35-ideas_l40s |
| Partition | gpu-l40s |
| GPU | NVIDIA L40S (1 per job) |
| Memory | 64 GB |
| Time | 20 hours per job |
| Conda Env | DPFM |

### Working Nodes
- atl1-1-03-007-29-0
- atl1-1-03-007-31-0
- atl1-1-01-010-29-0
- atl1-1-01-010-31-0
- atl1-1-01-010-33-0
- atl1-1-03-004-29-0

### Problematic Nodes (CUDA errors)
- atl1-1-03-004-31-0
- atl1-1-01-010-35-0

---

## Running Commands

### Check Job Status
```bash
# Check all running jobs
squeue -u eliu354 --format="%.10i %.15j %.2t %.10M %R"

# Check training progress
for job in ddpm_unet_s42_2526314 fm_unet_s42_2525656 fm_trans_s42_2525657 fm_steps4_2525658 fm_steps8_2525659 fm_steps16_2525660; do
    epoch=$(grep -E "Training epoch [0-9]+:" logs/experiments/${job}.err 2>/dev/null | tail -1 | grep -oP "epoch \K[0-9]+" | head -1)
    echo "$job: epoch ${epoch:-0} / 3050"
done

# Check specific job log
tail -30 logs/experiments/ddpm_unet_s42_2526314.err
```

### Submit Phase 2 Ablations (after Phase 1 completes)
```bash
bash scripts/submit_experiments.sh --ablation-lr
bash scripts/submit_experiments.sh --ablation-seeds
bash scripts/submit_experiments.sh --ablation-data
```

---

## Evaluation Commands

### Run Evaluation on Trained Model
```bash
python dpfm/eval.py \
    --checkpoint_path data/outputs/2025.12.04/<run_dir>/checkpoints/latest.ckpt \
    --n_eval_episodes 50 \
    --output_dir results/eval/<exp_name>
```

### Evaluate Best Previous Model
```bash
python dpfm/eval.py \
    --checkpoint_path data/outputs/2025.11.27/04.36.04_train_fm_unet_hybrid_pusht_image/checkpoints/epoch=0900-test_mean_score=0.845.ckpt \
    --n_eval_episodes 50 \
    --output_dir results/eval/fm_unet_best
```

---

## Metrics Tracked

| Metric | Description |
|--------|-------------|
| `test_mean_score` | Mean success score (0-1, higher is better) |
| `train_loss` | Training loss per step |
| `lr` | Learning rate per step |
| `coverage` | State space coverage metric |
| `success_rate` | Binary success rate |
| `avg_final_distance` | Average distance to goal at episode end |
| `avg_trajectory_smoothness` | Action smoothness metric |
| `avg_step_count` | Average steps per episode |
| `avg_inference_latency_ms` | Inference time per step |

---

## File Structure

```
dpfm/
├── config/
│   ├── task/pusht_image.yaml              # PushT task config
│   ├── train_ddpm_unet_hybrid_pusht.yaml  # DDPM baseline
│   ├── train_fm_unet_hybrid_image_workspace.yaml      # FM UNet
│   └── train_fm_transformer_hybrid_image_workspace.yaml  # FM Transformer
├── train.py                               # Training entry point
└── eval.py                                # Evaluation entry point

scripts/
├── submit_experiments.sh                  # Master experiment submission
├── submit_ddpm_baseline.sh               # DDPM baseline submission
└── test_ddpm_config.sh                   # Config validation

logs/experiments/
├── ddpm_unet_s42_2526314.{out,err}       # DDPM baseline logs
├── fm_unet_s42_2525656.{out,err}         # FM UNet logs
├── fm_trans_s42_2525657.{out,err}        # FM Transformer logs
├── fm_steps4_2525658.{out,err}           # FM 4 steps logs
├── fm_steps8_2525659.{out,err}           # FM 8 steps logs
└── fm_steps16_2525660.{out,err}          # FM 16 steps logs

data/outputs/
├── 2025.11.27/                           # Previous training runs
│   └── 04.36.04_train_fm_unet_hybrid_pusht_image/  # Best model (0.845)
└── 2025.12.04/                           # Current training runs
    ├── *_train_ddpm_unet_hybrid_pusht_image/
    ├── *_train_fm_unet_hybrid_pusht_image/
    └── *_train_fm_transformer_hybrid_pusht_image/
```

---

## Troubleshooting

### CUDA Errors
If you encounter `CUDA error: uncorrectable ECC error` or `CUDA-capable device(s) is/are busy`:
1. Check node allocation - some nodes have faulty GPUs
2. Reduce memory from 384GB to 64GB
3. Use known working nodes (see Cluster Configuration)

### Job Submission Issues
- Maximum ~6 jobs can run concurrently
- Use `--dry-run` flag to preview submissions
- Check job logs in `logs/experiments/`

---

## References

1. Chi et al. "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion" (2023)
2. Lipman et al. "Flow Matching for Generative Modeling" (2022)
3. Liu et al. "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow" (2022)

---

*Last updated: 2025-12-04 04:20 EST*
