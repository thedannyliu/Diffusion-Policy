# Flow Matching Ablation Study Design

This document outlines the experimental design for comparing Flow Matching (FM) against DDPM-based Diffusion Policy on the PushT benchmark.

## Research Questions

1. **How does Flow Matching compare to DDPM** in terms of success rate, inference speed, and training efficiency?
2. **What is the optimal number of inference steps** for Flow Matching to balance speed and quality?
3. **How sensitive is Flow Matching to learning rate** hyperparameters?
4. **Does the UNet or Transformer architecture** work better with Flow Matching?

## Baseline Configurations

### DDPM Baseline
Following the original Diffusion Policy paper settings:

| Setting | Value |
|---------|-------|
| Architecture | UNet Hybrid |
| Learning Rate | 1e-4 |
| Warmup Steps | 500 |
| Epochs | 3050 |
| Batch Size | 64 |
| Inference Steps | 100 (DDPM) |

### Flow Matching Baseline
FM configuration matching DDPM baseline:

| Setting | Value |
|---------|-------|
| Architecture | UNet Hybrid |
| Learning Rate | 1e-4 |
| Warmup Steps | 500 |
| Epochs | 3050 |
| Batch Size | 64 |
| Inference Steps | 4 (Euler) |

## Ablation Studies

### Ablation A: Inference Steps
**Goal**: Understand speed vs. quality trade-off in FM.

| Experiment | Steps | Rationale |
|------------|-------|-----------|
| FM-steps-4 | 4 | Baseline (fastest) |
| FM-steps-8 | 8 | Moderate speed/quality |
| FM-steps-16 | 16 | Higher quality |

### Ablation B: Learning Rate
**Goal**: Find optimal learning rate for FM.

| Experiment | Learning Rate | Rationale |
|------------|---------------|-----------|
| FM-lr1e-4 | 1e-4 | Baseline (matches DDPM) |
| FM-lr5e-5 | 5e-5 | Lower LR for stability |
| FM-lr1e-3 | 1e-3 | Higher LR for faster convergence |

### Ablation C: Architecture
**Goal**: Compare UNet vs Transformer backbones.

| Experiment | Architecture | Notes |
|------------|--------------|-------|
| FM-UNet | UNet Hybrid | Baseline |
| FM-Transformer | Transformer Hybrid | Alternative |

## Evaluation Metrics

1. **Test Mean Score**: Policy coverage metric (0-1, higher is better)
2. **Inference Latency**: p50 latency in milliseconds
3. **Training Time**: Wall-clock hours to complete training
4. **Smoothness**: Jerk-based trajectory smoothness

## Hardware Configuration

| Setting | Value |
|---------|-------|
| Cluster | Georgia Tech PACE Phoenix |
| GPU | NVIDIA L40S |
| Partition | gpu-l40s |
| Account | gts-agarg35-ideas_l40s |
| Memory | 64 GB per job |
| Time | 20 hours per job |

## Config Files

| Experiment | Config File |
|------------|-------------|
| DDPM Baseline | `dpfm/config/train_ddpm_unet_hybrid_pusht.yaml` |
| FM UNet | `dpfm/config/train_fm_unet_hybrid_image_workspace.yaml` |
| FM Transformer | `dpfm/config/train_fm_transformer_hybrid_image_workspace.yaml` |

## WandB Logging

All experiments logged to:
- **Project**: `dpfm_pusht_ablation`
- **Run Naming**: `{method}_{arch}_step{N}_seed{S}`

Example: `fm_unet_step4_seed42`
