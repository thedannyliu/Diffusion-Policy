# DPFM Experiments - Final Documentation

## Overview

This document contains the complete experimental setup for comparing **Diffusion Policy (DDPM)** and **Flow Matching (FM)** approaches on the PushT benchmark task.

## Currently Running Experiments (Phase 1: Baselines + Ablation A)

### Active Jobs (2025-12-04 04:10 EST)

| Job ID | Experiment | Method | Architecture | Inference Steps | Node | Status |
|--------|------------|--------|--------------|-----------------|------|--------|
| 2526314 | ddpm_unet_s42 | DDPM | UNet | 100 | atl1-1-03-004-29-0 | ✅ Training |
| 2525656 | fm_unet_s42 | FM | UNet | 4 | atl1-1-03-007-29-0 | ✅ Training |
| 2525657 | fm_trans_s42 | FM | Transformer | 4 | atl1-1-03-007-31-0 | ✅ Training |
| 2525658 | fm_steps4 | FM | UNet | 4 | atl1-1-01-010-29-0 | ✅ Training |
| 2525659 | fm_steps8 | FM | UNet | 8 | atl1-1-01-010-31-0 | ✅ Training |
| 2525660 | fm_steps16 | FM | UNet | 16 | atl1-1-01-010-33-0 | ✅ Training |

**Total: 6 jobs running (3 Baselines + 3 Ablation A)**

## Experiment Structure

### Phase 1: Baselines + Ablation A (Currently Running)

#### Baselines
| Experiment | Method | Architecture | Inference Steps | Epochs |
|------------|--------|--------------|-----------------|--------|
| `ddpm_unet_s42` | DDPM | UNet | 100 | 3050 |
| `fm_unet_s42` | FM | UNet | 4 | 3050 |
| `fm_trans_s42` | FM | Transformer | 4 | 3050 |

#### Ablation A: Inference Steps (FM only)
| Experiment | Inference Steps |
|------------|-----------------|
| `fm_steps4` | 4 steps (baseline) |
| `fm_steps8` | 8 steps |
| `fm_steps16` | 16 steps |

### Phase 2: Additional Ablations (To be submitted later)

#### Ablation B: Learning Rate + Warmup
- `fm_lr1e-4_w500`: lr=1e-4, warmup=500 (baseline)
- `fm_lr1e-4_w1000`: lr=1e-4, warmup=1000
- `fm_lr2e-4_w500`: lr=2e-4, warmup=500
- `fm_lr2e-4_w1000`: lr=2e-4, warmup=1000

#### Ablation C: Seed Sensitivity
- Seeds: 42, 43, 44, 45, 46 (5 runs)

#### Ablation D: Dataset Size
- `fm_data90`, `ddpm_data90`: 90 episodes (full)
- `fm_data60`, `ddpm_data60`: 60 episodes
- `fm_data30`, `ddpm_data30`: 30 episodes

## Metrics Tracked

All experiments log the following to WandB:

| Metric | Description |
|--------|-------------|
| `test_mean_score` | Mean success score (0-1) |
| `train_loss` | Training loss per step |
| `lr` | Learning rate per step |
| `coverage` | State space coverage metric |
| `success_rate` | Binary success rate |
| `avg_final_distance` | Avg distance to goal at episode end |
| `avg_trajectory_smoothness` | Action smoothness metric |
| `avg_step_count` | Avg steps per episode |
| `avg_inference_latency_ms` | Inference time per step |

## Cluster Configuration

- **Cluster**: Georgia Tech PACE Phoenix
- **Account**: gts-agarg35-ideas_l40s
- **Partition**: gpu-l40s
- **GPU**: NVIDIA L40S (1 per job)
- **Memory**: 64 GB (reduced from 384GB to allow multiple jobs per node)
- **Time**: 20 hours per job
- **Conda Environment**: DPFM

### Node Information
- Each node has 8 L40S GPUs and 515GB memory
- Multiple jobs can run on the same node with 64GB memory allocation
- Known working nodes: atl1-1-03-007-29-0, atl1-1-03-007-31-0, atl1-1-01-010-29-0, atl1-1-01-010-31-0, atl1-1-01-010-33-0, atl1-1-03-004-29-0
- Problematic nodes (CUDA errors): atl1-1-03-004-31-0, atl1-1-01-010-35-0

## WandB Project

- **Project**: `dpfm_pusht_experiments`
- **URL**: https://wandb.ai/danny010324/dpfm_pusht_experiments
- **Groups**:
  - `ddpm_baselines`
  - `fm_baselines`
  - `ablation_steps`
  - `ablation_lr`
  - `ablation_seeds`
  - `ablation_data`

## File Structure

```
dpfm/
├── config/
│   ├── task/
│   │   └── pusht_image.yaml                        # PushT task config
│   ├── train_ddpm_unet_hybrid_pusht.yaml           # DDPM baseline config
│   ├── train_fm_unet_hybrid_image_workspace.yaml   # FM UNet config
│   └── train_fm_transformer_hybrid_image_workspace.yaml  # FM Transformer config
├── policy/
│   ├── flow_matching_unet_hybrid_image_policy.py
│   └── flow_matching_transformer_hybrid_image_policy.py
├── workspace/
│   ├── train_fm_unet_hybrid_image_workspace.py
│   └── train_fm_transformer_hybrid_image_workspace.py
└── train.py

scripts/
├── submit_experiments.sh             # Master experiment submission
├── submit_ddpm_baseline.sh           # DDPM baseline submission
└── test_ddpm_config.sh              # DDPM config validation

docs/
├── ablation_design.md               # Ablation study design
└── final.md                         # This file

diffusion_policy/
└── env/pusht/pusht_visualization.py # Heatmap trajectory plots
```

## Previous Training Results

Existing trained models from earlier experiments:

| Path | Best Score | Notes |
|------|------------|-------|
| `data/outputs/2025.11.27/04.36.04_train_fm_unet_hybrid_pusht_image/` | 0.845 | FM UNet, epoch 900 |
| `data/outputs/2025.11.27/14.28.45_train_fm_unet_hybrid_pusht_image/` | 0.838 | FM UNet, epoch 200 |
| `data/outputs/2025.11.29/00.32.47_train_fm_unet_hybrid_pusht_image/` | 0.823 | FM UNet, epoch 650 |

## Running Commands

### Check Job Status
```bash
# Check all running jobs
squeue -u eliu354 --format="%.10i %.15j %.2t %.10M %R"

# Check specific job progress
tail -30 logs/experiments/ddpm_unet_s42_2526314.err
grep "Training epoch" logs/experiments/fm_unet_s42_2525656.err | tail -5
```

### Submit New Jobs
```bash
# Submit remaining ablations after Phase 1 completes
bash scripts/submit_experiments.sh --ablation-lr
bash scripts/submit_experiments.sh --ablation-seeds
bash scripts/submit_experiments.sh --ablation-data
```

## Evaluation (After Training Completes)

### Run Evaluation
```bash
# Evaluate a trained model
python dpfm/eval.py \
    --checkpoint_path data/outputs/2025.12.04/<run_dir>/checkpoints/latest.ckpt \
    --n_eval_episodes 50 \
    --output_dir results/eval/<exp_name>
```

### Generate Trajectory Heatmaps
The `PushTImageRunner` automatically generates heatmap-style trajectory visualizations similar to the original Diffusion Policy paper figure during evaluation.

## Troubleshooting

### CUDA Errors
If you encounter `CUDA error: uncorrectable ECC error` or `CUDA-capable device(s) is/are busy or unavailable`:
1. Check node allocation - some nodes have faulty GPUs
2. Use known working nodes (see Cluster Configuration section)
3. Reduce memory allocation from 384GB to 64GB

### Job Submission Issues
- Maximum ~6 jobs can run concurrently
- Use `--dry-run` flag to preview submissions
- Check job logs in `logs/experiments/`

## References

1. Chi et al. "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion" (2023)
2. Lipman et al. "Flow Matching for Generative Modeling" (2022)
3. Liu et al. "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow" (2022)

---

*Last updated: 2025-12-04 04:10 EST*
