# DPFM Experiments - Final Documentation

## Overview

This document contains the complete experimental setup for comparing **Diffusion Policy (DDPM)** and **Flow Matching (FM)** approaches on the PushT benchmark task.

## Experiment Structure

### 1. Baseline Experiments

| Experiment | Method | Architecture | Inference Steps | Epochs |
|------------|--------|--------------|-----------------|--------|
| `ddpm_unet_s42` | DDPM | UNet | 100 | 3050 |
| `fm_unet_s42` | FM | UNet | 4 | 3050 |
| `fm_trans_s42` | FM | Transformer | 4 | 3050 |

### 2. Ablation Studies

#### Ablation A: Inference Steps (FM only)
- `fm_steps4`: 4 inference steps (baseline)
- `fm_steps8`: 8 inference steps
- `fm_steps16`: 16 inference steps

#### Ablation B: Learning Rate + Warmup
- `fm_lr1e-4_w500`: lr=1e-4, warmup=500 (baseline)
- `fm_lr1e-4_w1000`: lr=1e-4, warmup=1000
- `fm_lr2e-4_w500`: lr=2e-4, warmup=500
- `fm_lr2e-4_w1000`: lr=2e-4, warmup=1000

#### Ablation C: Seed Sensitivity
- Seeds: 42, 43, 44, 45, 46 (5 runs)

#### Ablation D: Dataset Size
- `fm_data90`: 90 episodes (full)
- `fm_data60`: 60 episodes
- `fm_data30`: 30 episodes
- `ddpm_data90`: 90 episodes (full)
- `ddpm_data60`: 60 episodes
- `ddpm_data30`: 30 episodes

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
- **GPU**: NVIDIA L40S (1 per job)
- **Memory**: 384 GB
- **Time**: 20 hours per job
- **Conda**: DPFM

## File Structure

```
dpfm/
├── config/
│   ├── task/
│   │   └── pusht_image.yaml          # PushT task config
│   ├── train_ddpm_unet_hybrid_pusht.yaml    # DDPM baseline
│   ├── train_fm_unet_hybrid_image_workspace.yaml    # FM UNet
│   └── train_fm_transformer_hybrid_image_workspace.yaml  # FM Transformer
├── policy/
│   ├── flow_matching_unet_hybrid_image_policy.py
│   └── flow_matching_transformer_hybrid_image_policy.py
├── workspace/
│   ├── train_fm_unet_hybrid_image_workspace.py
│   └── train_fm_transformer_hybrid_image_workspace.py
└── train.py

scripts/
├── submit_experiments.sh             # Master experiment submission
└── test_ddpm_config.sh              # DDPM config test

docs/
├── ablation_design.md               # Ablation study design
└── final.md                         # This file

diffusion_policy/
└── env/pusht/pusht_visualization.py # Heatmap trajectory plots
```

## Running Experiments

### Test Single Config
```bash
# Test DDPM config (2 epochs, no wandb)
sbatch scripts/test_ddpm_config.sh
```

### Submit All Experiments
```bash
# Dry run to see what will be submitted
bash scripts/submit_experiments.sh --dry-run

# Submit all baselines
bash scripts/submit_experiments.sh --baseline

# Submit specific ablation
bash scripts/submit_experiments.sh --ablation-steps
bash scripts/submit_experiments.sh --ablation-lr
bash scripts/submit_experiments.sh --ablation-seeds
bash scripts/submit_experiments.sh --ablation-data

# Submit everything
bash scripts/submit_experiments.sh
```

## Job Tracking

### Submitted Jobs

| Job ID | Experiment | Status | Notes |
|--------|------------|--------|-------|
| 2525610 | test_ddpm | Pending | Config validation test |
| TBD | All experiments | - | - |

## WandB Project

- **Project**: `dpfm_pusht_experiments`
- **Groups**:
  - `ddpm_baselines`
  - `fm_baselines`
  - `ablation_steps`
  - `ablation_lr`
  - `ablation_seeds`
  - `ablation_data`

## Key Findings

*To be filled after experiments complete.*

### Expected Results

Based on prior work:
- FM should achieve similar success rate to DDPM
- FM with 4-16 steps should be 6-25x faster than DDPM (100 steps)
- FM should maintain trajectory quality with fewer steps

### Trajectory Visualization

Each evaluation generates:
1. Heatmap-style trajectory plot (multiple rollouts overlaid)
2. T-shape goal region clearly visible
3. Color gradient showing trajectory progression
4. Coverage and success statistics

## References

1. Chi et al. "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion" (2023)
2. Lipman et al. "Flow Matching for Generative Modeling" (2022)
3. Liu et al. "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow" (2022)

---

*Last updated: [Date]*
*Author: [Your Name]*
