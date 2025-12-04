# Flow Matching Ablation Study Design

This document outlines the ablation experiments for Flow Matching (FM) policy on PushT benchmark.

## Overview

We conduct systematic ablation studies to understand the key factors affecting Flow Matching performance and compare against DDPM baselines from the original Diffusion Policy paper.

## Baseline Configurations

### DDPM Baselines (from Original Paper)
Based on the original Diffusion Policy paper settings:

| Architecture | Learning Rate | Warmup Steps | Epochs | Batch Size | Inference Steps |
|--------------|---------------|--------------|--------|------------|-----------------|
| UNet Hybrid | 1e-4 | 500 | 3050 | 64 | 100 (DDPM) |
| Transformer Hybrid | 1e-4 | 500 | 3050 | 64 | 100 (DDPM) |

### Flow Matching Baselines
FM configurations matching DDPM baselines:

| Architecture | Learning Rate | Warmup Steps | Epochs | Batch Size | Inference Steps |
|--------------|---------------|--------------|--------|------------|-----------------|
| FM UNet Hybrid | 1e-4 | 500 | 3050 | 64 | 4 (Euler) |
| FM Transformer Hybrid | 1e-4 | 500 | 3050 | 64 | 4 (Euler) |

## Ablation Studies

### A. Inference Steps (FM-specific)
**Goal**: Understand speed vs. quality trade-off in FM.

| Experiment | Inference Steps | Expected Latency |
|------------|-----------------|------------------|
| FM-steps-4 | 4 | ~20-30ms |
| FM-steps-8 | 8 | ~40-60ms |
| FM-steps-16 | 16 | ~80-120ms |

**Metrics**: Success rate, coverage, inference latency

### B. Learning Rate and Warmup
**Goal**: Find optimal training stability vs. convergence speed.

| Experiment | Learning Rate | Warmup Steps |
|------------|---------------|--------------|
| FM-lr1e4-w500 | 1e-4 | 500 |
| FM-lr1e4-w1000 | 1e-4 | 1000 |
| FM-lr2e4-w500 | 2e-4 | 500 |
| FM-lr2e4-w1000 | 2e-4 | 1000 |

**Metrics**: Training loss curve, final success rate, training stability

### C. Seed Sensitivity
**Goal**: Report robust mean/std statistics.

| Experiment | Seed |
|------------|------|
| FM-seed-42 | 42 |
| FM-seed-43 | 43 |
| FM-seed-44 | 44 |
| FM-seed-45 | 45 |
| FM-seed-46 | 46 |

**Metrics**: Mean ± std of success rate, coverage

### D. Dataset Size (Data Efficiency)
**Goal**: Compare FM and DDPM data efficiency.

| Experiment | Max Episodes | % of Full Data |
|------------|--------------|----------------|
| *-data-90 | 90 | 100% |
| *-data-60 | 60 | 67% |
| *-data-30 | 30 | 33% |

Run for both FM and DDPM to compare.

**Metrics**: Success rate, coverage at each data size

## Evaluation Metrics

All experiments report:

1. **Success Rate**: Coverage ≥ 0.95
2. **Target-Area Coverage**: Maximum overlap ratio with goal
3. **Final Distance**: L2 distance from block to goal at episode end
4. **Smoothness**: Jerk-based metric on block trajectory
5. **Step Count**: Number of steps to reach goal
6. **Inference Latency**: p50, p95, mean (ms)
7. **Training Time**: Total wall-clock time

## Hardware Configuration

- GPU: NVIDIA L40S
- Account: gts-agarg35-ideas_l40s
- Partition: gpu-l40s
- Memory: 384GB
- Time: 20 hours per job

## File Naming Convention

Training configs:
- `train_ddpm_unet_pusht.yaml` - DDPM UNet baseline
- `train_ddpm_transformer_pusht.yaml` - DDPM Transformer baseline
- `train_fm_unet_pusht.yaml` - FM UNet baseline
- `train_fm_transformer_pusht.yaml` - FM Transformer baseline

Ablation configs follow pattern:
- `train_fm_unet_pusht_steps{N}.yaml` - Inference steps ablation
- `train_fm_unet_pusht_lr{X}_w{Y}.yaml` - LR/warmup ablation
- `train_fm_unet_pusht_seed{N}.yaml` - Seed ablation
- `train_{method}_unet_pusht_data{N}.yaml` - Data size ablation

## Expected Outcomes

1. **FM vs DDPM**: FM should achieve similar success rate with 4-10x faster inference
2. **Inference Steps**: FM with 4 steps should be near-optimal; 8-16 may give slight improvements
3. **Data Efficiency**: FM may show better data efficiency due to simpler learning objective
4. **Architecture**: UNet and Transformer should have similar performance, Transformer may be slower

## WandB Logging

All experiments logged to WandB with:
- Project: `diffusion-policy-fm`
- Entity: (as configured)
- Group: `ablation-{experiment_type}`

Tags:
- `baseline` / `ablation`
- `ddpm` / `fm`
- `unet` / `transformer`
- `pusht`
