# Experimental Results: Flow Matching vs DDPM for Diffusion Policy

This document presents the comprehensive results of our ablation study comparing Flow Matching (FM) against DDPM for visuomotor policy learning on the PushT benchmark.

## Executive Summary

| Experiment | Method | Arch | Steps | Val Score | Test Score | Latency (p50) | Training Time | Speedup |
|------------|--------|------|-------|-----------|------------|---------------|---------------|---------|
| **ddpm_unet_s42** | DDPM | UNet | 100 | **0.869** | **0.816** | 635.0 ms | 19.4 hr | 1.0× |
| fm_steps16 | FM | UNet | 16 | 0.844 | 0.794 | 90.4 ms | 8.9 hr | **7.0×** |
| fm_lr5e-5 | FM | UNet | 4 | 0.820 | 0.798 | 23.1 ms | 6.8 hr | **27.5×** |
| fm_steps8 | FM | UNet | 8 | 0.801 | 0.769 | 45.3 ms | 8.0 hr | **14.0×** |
| fm_unet_s42 | FM | UNet | 4 | 0.777 | 0.667 | 23.1 ms | 6.5 hr | **27.5×** |
| fm_lr1e-3 | FM | UNet | 4 | 0.741 | 0.666 | 23.1 ms | 6.7 hr | **27.5×** |
| fm_trans_s42 | FM | Trans | 4 | 0.117 | 0.114 | 20.9 ms | 7.2 hr | **30.4×** |

**Key Findings**:
- FM with optimized LR (5e-5) achieves **98% of DDPM test performance** with **27× faster inference**
- FM trains **2-3× faster** than DDPM (6-9 hours vs 19 hours)
- Transformer architecture fails with FM; UNet is essential

---

## Training Phase Results

All models were trained for 3050 epochs on the PushT dataset with identical hyperparameters except for the ablated variable.

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 3050 |
| Batch Size | 64 |
| Optimizer | AdamW |
| LR Schedule | Cosine with warmup |
| Warmup Steps | 500 |
| EMA | Enabled (decay=0.9999) |
| GPU | NVIDIA L40S (48GB) |

### Training Time Comparison

| Experiment | Wall-clock Time | GPU Hours | Relative Speed |
|------------|-----------------|-----------|----------------|
| ddpm_unet_s42 | 19:22:20 | 19.4 hr | 1.0× |
| fm_steps16 | 08:51:29 | 8.9 hr | 2.2× |
| fm_steps8 | 07:59:32 | 8.0 hr | 2.4× |
| fm_trans_s42 | 07:09:27 | 7.2 hr | 2.7× |
| fm_lr5e-5 | 06:47:24 | 6.8 hr | 2.9× |
| fm_lr1e-3 | 06:44:50 | 6.7 hr | 2.9× |
| fm_steps4 | 06:46:19 | 6.8 hr | 2.9× |
| fm_unet_s42 | 06:27:52 | 6.5 hr | 3.0× |

**Observation**: FM models train 2-3× faster than DDPM due to simpler loss computation and fewer inference steps during validation rollouts.

---

## Validation Phase Results

Validation was performed every 50 epochs using rollouts in the PushT environment. The best checkpoint was selected based on validation (test_mean_score during training).

### Best Validation Scores

| Experiment | Best Epoch | Val Score | Val Coverage | Checkpoint |
|------------|------------|-----------|--------------|------------|
| ddpm_unet_s42 | 450 | **0.869** | 0.832 | epoch=0450-test_mean_score=0.869.ckpt |
| fm_steps16 | 150 | 0.844 | 0.807 | epoch=0150-test_mean_score=0.844.ckpt |
| fm_lr5e-5 | 450 | 0.820 | 0.784 | epoch=0450-test_mean_score=0.820.ckpt |
| fm_steps8 | 150 | 0.801 | 0.765 | epoch=0150-test_mean_score=0.801.ckpt |
| fm_unet_s42 | 150 | 0.777 | 0.743 | epoch=0150-test_mean_score=0.777.ckpt |
| fm_lr1e-3 | 600 | 0.741 | 0.708 | epoch=0600-test_mean_score=0.741.ckpt |
| fm_trans_s42 | 0 | 0.117 | 0.112 | epoch=0000-test_mean_score=0.117.ckpt |

**Observation**: 
- DDPM reaches best performance at epoch 450
- FM models with default LR converge faster (epoch 150), while lower LR (5e-5) needs more epochs (450)
- Transformer fails to learn, with best score at epoch 0 (random initialization level)

---

## Test Phase Results

Final evaluation was performed on 50 test episodes using the best checkpoint from each experiment.

### Test Metrics (50 Episodes)

| Experiment | Test Score | Success Rate | Coverage | Final Dist | Steps | Smoothness |
|------------|------------|--------------|----------|------------|-------|------------|
| ddpm_unet_s42 | **0.816** | 100% | 0.781 | 30.6 | 238.4 | 0.93 |
| fm_lr5e-5 | 0.798 | 100% | 0.761 | 49.8 | 257.0 | 0.94 |
| fm_steps16 | 0.794 | 100% | 0.758 | 56.1 | 242.4 | 0.97 |
| fm_steps8 | 0.769 | 100% | 0.735 | 60.2 | 243.9 | 1.51 |
| fm_unet_s42 | 0.667 | 100% | 0.637 | 79.2 | 278.0 | 1.40 |
| fm_lr1e-3 | 0.666 | 100% | 0.637 | 70.3 | 260.8 | 0.94 |
| fm_trans_s42 | 0.114 | 100% | 0.108 | 226.6 | 300.0 | 39.57 |

**Metrics Explanation**:
- **Test Score**: Overall performance metric (0-1, higher is better)
- **Success Rate**: Percentage of episodes completing without failure
- **Coverage**: Fraction of target area covered by the T-block
- **Final Dist**: Distance to target at episode end (lower is better)
- **Steps**: Average steps per episode (max 300)
- **Smoothness**: Trajectory jerk (lower is smoother)

### Inference Latency (50 Episodes)

| Experiment | Mean (ms) | p50 (ms) | p95 (ms) | Speedup vs DDPM |
|------------|-----------|----------|----------|-----------------|
| ddpm_unet_s42 | 637.0 | 635.0 | 637.5 | 1.0× |
| fm_steps16 | 92.2 | 90.4 | 90.6 | **7.0×** |
| fm_steps8 | 49.0 | 45.3 | 45.4 | **14.0×** |
| fm_lr5e-5 | 24.9 | 23.1 | 23.2 | **27.5×** |
| fm_unet_s42 | 24.9 | 23.1 | 23.2 | **27.5×** |
| fm_lr1e-3 | 24.9 | 23.1 | 23.2 | **27.5×** |
| fm_trans_s42 | 57.7 | 20.9 | 21.0 | **30.4×** |

**Observation**: FM with 4 steps achieves ~23ms latency, suitable for real-time control at 40+ Hz.

---

## Ablation Analysis

### Ablation A: Number of Inference Steps

**Question**: How many sampling steps does FM need for good performance?

| Steps | Val Score | Test Score | Latency (p50) | Speedup |
|-------|-----------|------------|---------------|---------|
| 4 | 0.777 | 0.667 | 23.1 ms | 27.5× |
| 8 | 0.801 | 0.769 | 45.3 ms | 14.0× |
| 16 | 0.844 | 0.794 | 90.4 ms | 7.0× |
| 100 (DDPM) | 0.869 | 0.816 | 635.0 ms | 1.0× |

**Conclusion**: 
- 16 steps achieves 97% of DDPM validation performance
- Diminishing returns beyond 8 steps for test performance
- For real-time applications (>30 Hz), use 4 steps

### Ablation B: Learning Rate

**Question**: What is the optimal learning rate for FM?

| Learning Rate | Val Score | Test Score | Best Epoch | Notes |
|---------------|-----------|------------|------------|-------|
| 1e-4 (default) | 0.777 | 0.667 | 150 | Fast convergence, lower final |
| **5e-5** | **0.820** | **0.798** | 450 | Best overall, more stable |
| 1e-3 | 0.741 | 0.666 | 600 | Too aggressive, unstable |

**Conclusion**: 
- FM benefits from lower LR (5e-5) compared to DDPM default (1e-4)
- Lower LR improves both validation and test scores by ~6%
- Higher LR (1e-3) causes unstable training with worse final performance

### Ablation C: Architecture

**Question**: Does Transformer work as well as UNet for FM?

| Architecture | Val Score | Test Score | Latency | Status |
|--------------|-----------|------------|---------|--------|
| **UNet Hybrid** | 0.777+ | 0.667+ | 23.1 ms | ✅ Works |
| Transformer Hybrid | 0.117 | 0.114 | 20.9 ms | ❌ Failed |

**Conclusion**: 
- Transformer completely fails to learn with FM
- UNet's inductive biases (spatial hierarchies, skip connections) are crucial
- This differs from some FM image generation works where Transformer succeeds

---

## Validation vs Test Performance Gap

An important observation is the gap between validation and test scores:

| Experiment | Val Score | Test Score | Gap |
|------------|-----------|------------|-----|
| ddpm_unet_s42 | 0.869 | 0.816 | -6.1% |
| fm_steps16 | 0.844 | 0.794 | -5.9% |
| fm_lr5e-5 | 0.820 | 0.798 | -2.7% |
| fm_steps8 | 0.801 | 0.769 | -4.0% |
| fm_unet_s42 | 0.777 | 0.667 | -14.2% |
| fm_lr1e-3 | 0.741 | 0.666 | -10.1% |

**Analysis**:
- FM with lower LR (5e-5) shows **smallest generalization gap** (2.7%)
- Default FM (1e-4) shows larger gap (14.2%), suggesting overfitting
- DDPM shows moderate gap (6.1%), indicating good regularization

---

## Conclusions

1. **Flow Matching achieves competitive performance**: FM-lr5e-5 reaches 98% of DDPM test performance while being 27× faster

2. **Learning rate is critical for FM**: Lower LR (5e-5) significantly improves both performance and generalization

3. **4-8 inference steps are practical**: 4 steps enables real-time control (40+ Hz), 8-16 steps for higher quality

4. **UNet architecture is essential**: Transformer fails completely with FM on this task

5. **FM trains 2-3× faster**: Reduced training time from 19 hours (DDPM) to 6-9 hours (FM)

6. **Trade-off recommendation**:
   - **Real-time applications**: FM with 4 steps, lr=5e-5 (23ms, 0.798 score)
   - **Best quality**: FM with 16 steps (90ms, 0.794 score) or DDPM (635ms, 0.816 score)

---

## Training Logs Reference

| Experiment | Job ID | Log Files |
|------------|--------|-----------|
| ddpm_unet_s42 | 2526314 | `logs/experiments/ddpm_unet_s42_2526314.{out,err}` |
| fm_unet_s42 | 2525656 | `logs/experiments/fm_unet_s42_2525656.{out,err}` |
| fm_trans_s42 | 2525657 | `logs/experiments/fm_trans_s42_2525657.{out,err}` |
| fm_steps4 | 2525658 | `logs/experiments/fm_steps4_2525658.{out,err}` |
| fm_steps8 | 2525659 | `logs/experiments/fm_steps8_2525659.{out,err}` |
| fm_steps16 | 2525660 | `logs/experiments/fm_steps16_2525660.{out,err}` |
| fm_lr5e-5 | 2528840 | `logs/experiments/fm_lr5e-5_2528840.{out,err}` |
| fm_lr1e-3 | 2533021 | `logs/experiments/fm_lr1e-3_2533021.{out,err}` |

---

## Evaluation Results Reference

Full evaluation results are stored in `results/` directory:

| Experiment | Results Directory |
|------------|-------------------|
| ddpm_unet_s42 | `results/ddpm_unet_s42/` |
| fm_steps16 | `results/fm_steps16/` |
| fm_lr5e-5 | `results/fm_lr5e-5/` |
| fm_steps8 | `results/fm_steps8/` |
| fm_unet_s42 | `results/fm_unet_s42/` |
| fm_lr1e-3 | `results/fm_lr1e-3/` |
| fm_trans_s42 | `results/fm_trans_s42/` |

Each directory contains:
- `eval_results.json`: Summary metrics
- `eval_log_full.json`: Per-episode detailed logs
