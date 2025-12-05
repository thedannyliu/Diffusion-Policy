# Experimental Results: Flow Matching vs DDPM for Diffusion Policy

This document presents the final results of our ablation study comparing Flow Matching (FM) against DDPM for visuomotor policy learning on the PushT benchmark.

## Summary of Results

### Main Findings

| Experiment | Method | Steps | Best Score | Latency (p50) | Training Time | Speedup |
|------------|--------|-------|------------|---------------|---------------|---------|
| **ddpm_unet_s42** | DDPM | 100 | **0.869** | 650.0 ms | 19.4 hr | 1.0× |
| fm_steps16 | FM | 16 | 0.844 | 150.7 ms | 8.9 hr | **4.3×** |
| fm_lr5e-5 | FM | 4 | 0.820 | 24.5 ms | 6.8 hr | **26.5×** |
| fm_steps8 | FM | 8 | 0.801 | 48.0 ms | 8.0 hr | **13.5×** |
| fm_unet_s42 | FM | 4 | 0.777 | 24.5 ms | 6.5 hr | **26.5×** |
| fm_lr1e-3 | FM | 4 | 0.741 | 24.5 ms | 6.7 hr | **26.5×** |
| fm_trans_s42 | FM Trans | 4 | 0.117 | 21.1 ms | 7.2 hr | **30.8×** |

### Key Insights

1. **DDPM Baseline**: Achieves best absolute score (0.869) but with highest latency (650ms)
2. **Best FM Configuration**: FM with 16 steps achieves 0.844 (97% of DDPM) with 4.3× speedup
3. **Best Speed/Quality Trade-off**: FM with LR=5e-5 achieves 0.820 with 26.5× speedup
4. **Optimal Steps**: 8-16 steps provide best balance; 4 steps is fastest but lower quality
5. **Architecture**: Transformer fails with FM (0.117), UNet is strongly preferred
6. **Training Efficiency**: FM trains 2-3× faster than DDPM

---

## Detailed Results

### Ablation A: Inference Steps

| Steps | Best Score | Best Epoch | p50 Latency | Speedup vs DDPM |
|-------|------------|------------|-------------|-----------------|
| 4 | 0.777 | 150 | 24.5 ms | 26.5× |
| 8 | 0.801 | 150 | 48.0 ms | 13.5× |
| 16 | 0.844 | 150 | 150.7 ms | 4.3× |

**Observation**: More inference steps improve quality with diminishing returns. 16 steps achieves near-DDPM quality.

### Ablation B: Learning Rate

| Learning Rate | Best Score | Best Epoch | Notes |
|---------------|------------|------------|-------|
| 1e-4 (baseline) | 0.777 | 150 | Default setting |
| 5e-5 | **0.820** | 450 | Best FM result with 4 steps |
| 1e-3 | 0.741 | 600 | Too aggressive, lower score |

**Observation**: Lower learning rate (5e-5) significantly improves FM performance, suggesting FM benefits from more stable training.

### Ablation C: Architecture

| Architecture | Best Score | Latency | Status |
|--------------|------------|---------|--------|
| UNet Hybrid | 0.777+ | 24.5 ms | ✅ Works well |
| Transformer Hybrid | 0.117 | 21.1 ms | ❌ Failed |

**Observation**: Transformer architecture fails to learn meaningful policies with Flow Matching on PushT task.

---

## Training Logs

SLURM job logs are available in `logs/experiments/`:

| Experiment | Job ID | Log Files |
|------------|--------|-----------|
| ddpm_unet_s42 | 2526314 | `ddpm_unet_s42_2526314.{out,err}` |
| fm_unet_s42 | 2525656 | `fm_unet_s42_2525656.{out,err}` |
| fm_trans_s42 | 2525657 | `fm_trans_s42_2525657.{out,err}` |
| fm_steps4 | 2525658 | `fm_steps4_2525658.{out,err}` |
| fm_steps8 | 2525659 | `fm_steps8_2525659.{out,err}` |
| fm_steps16 | 2525660 | `fm_steps16_2525660.{out,err}` |
| fm_lr5e-5 | 2528840 | `fm_lr5e-5_2528840.{out,err}` |
| fm_lr1e-3 | 2533021 | `fm_lr1e-3_2533021.{out,err}` |

---

## Conclusions

1. **Flow Matching is a viable alternative to DDPM** for diffusion policy, achieving 97% of DDPM performance with 4× speedup
2. **Lower learning rates work better for FM** (5e-5 vs 1e-4)
3. **4-8 steps is optimal for FM** when prioritizing speed; 16 steps for best quality
4. **UNet architecture is essential**; Transformer fails with FM on this task
5. **Training is significantly faster** with FM (6-9 hours vs 19 hours)
