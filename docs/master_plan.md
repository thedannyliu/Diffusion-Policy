# Diffusion Policy with Flow Matching (DPFM) - Master Development Plan

> **Project Goal**: Replace DDPM sampling in Diffusion Policy with Flow Matching to achieve comparable task success with significantly fewer sampling steps, reducing inference latency for robotic control.

> **Due Date**: December 5, 2025

> **Last Updated**: November 26, 2025

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Background & Motivation](#2-background--motivation)
3. [Technical Approach](#3-technical-approach)
4. [Project Structure](#4-project-structure)
5. [Implementation Plan](#5-implementation-plan)
6. [Experiment Design](#6-experiment-design)
7. [Timeline](#7-timeline)
8. [Risk Assessment & Fallback](#8-risk-assessment--fallback)
9. [Deliverables](#9-deliverables)
10. [References](#10-references)

---

## 1. Project Overview

### 1.1 Research Question

> **Can Flow Matching training enable Diffusion Policy to achieve comparable success rates with 1-4 sampling steps, while significantly reducing control latency and action jerk?**

### 1.2 Core Hypothesis

Flow Matching learns a direct ODE path from noise to data, enabling high-quality action generation with far fewer inference steps than DDPM's iterative denoising process.

### 1.3 Key Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Success Rate | Task completion % | Match or exceed DDPM baseline |
| Inference Latency | p50/p95 per control step | 2-5x reduction |
| Action Jerk | Mean ||a_t - a_{t-1}||² | Equal or lower |
| Sampling Steps | Number of ODE steps | 1, 2, 4 steps vs 100 DDPM |

---

## 2. Background & Motivation

### 2.1 Diffusion Policy Pain Point

Standard DDPM in Diffusion Policy requires **100 denoising steps** at inference, causing:

- **High latency**: 50-100ms per control step on GPU
- **Low control frequency**: Limits real-time closed-loop control
- **Action jitter**: Many noisy updates can cause oscillations

### 2.2 Flow Matching Solution

Flow Matching provides an alternative generative framework:

- **Direct ODE path**: Learns vector field for straight-line interpolation
- **Few-step inference**: Euler integration with 1-4 steps
- **Simple loss**: MSE between predicted and target velocity

### 2.3 Mathematical Formulation

**DDPM (current, original Diffusion Policy)**:
```
Forward: x_t = √(α_t) * x_0 + √(1-α_t) * ε
Loss: L = ||ε_θ(x_t, t) - ε||²
Inference: x_{t-1} = f(x_t, ε_θ(x_t, t))  [100 steps]
```

**Flow Matching / Rectified Flow-style (proposed)**:
```
Interpolation: x_t = t * x_1 + (1-t) * x_0  [x_1 = data, x_0 = noise]
Target velocity: u_t = x_1 - x_0
Loss: L = ||v_θ(x_t, t, obs) - u_t||²
Inference: x_1 = x_0 + Σ v_θ(x_t, t) * Δt  [1-4 steps]
```

**Design note (literature alignment)**:
- This objective corresponds to the *linear optimal-transport path* often used in **Rectified Flow** (Liu et al., 2023) and **Flow Matching** variants: we transport a standard Gaussian `x_0 ~ N(0, I)` to the data `x_1` along a straight line in data space.  
- We intentionally use the *simplified* version without explicit density correction terms or path reweighting to keep the modification to Diffusion Policy minimal and stable for RL control. The comparison to DDPM is therefore: *same UNet, same conditioning, same data distribution, only the training objective and sampler change*.

---

## 3. Technical Approach

### 3.1 What We Keep Unchanged

| Component | File | Reason |
|-----------|------|--------|
| Visual Encoder | `multi_image_obs_encoder.py` | Isolate the experiment variable |
| UNet Backbone | `conditional_unet1d.py` | Same architecture, different objective |
| Data Pipeline | `pusht_image_dataset.py` | Same data and preprocessing as original DP Push-T |
| Normalizer | `normalizer.py` | Same preprocessing |

### 3.2 What We Modify

| Component | Original | Modified |
|-----------|----------|----------|
| Training Loss | DDPM noise prediction | Flow Matching velocity prediction |
| Sampling | DDPM reverse diffusion (100 steps) | Euler ODE integration (1-4 steps) |
| Time Embedding | Discrete timesteps | Continuous t ∈ [0, 1] |

For fairness to the original paper and codebase:
- **Environment, dataset, and observation space**: identical to the official Push-T image setting in Diffusion Policy (same zarr dataset, same image resolution, same history length).
- **Model architecture**: we reuse `ConditionalUnet1D` with the same `down_dims`, kernel size, group norm, and visual encoder (`MultiImageObsEncoder`).
- **Training pipeline**: we keep the same optimizer, batch size, learning rate schedule, EMA, and rollout/evaluation schedule as `TrainDiffusionUnetImageWorkspace`; only the policy type and its loss/sampling internals differ.

### 3.3 Flow Matching Loss Implementation

We implement Flow Matching as a *Rectified Flow-style* loss that is compatible with the existing Diffusion Policy training structure:

- Input to the loss:
  - `model`: the same `ConditionalUnet1D` used in DDPM.  
  - `x_1`: **normalized** action sequences `[B, T, D]` (same normalization as DDPM, via `LinearNormalizer`).  
  - `global_cond`: observation features `[B, cond_dim]` obtained with `MultiImageObsEncoder` in exactly the same way as in `DiffusionUnetImagePolicy.compute_loss`.

- Loss definition (pseudo-code):
  ```python
  # Sample time and base noise
  t ~ Uniform(0, 1)          # shape [B]
  x0 ~ N(0, I)               # same shape as x1

  # Linear OT path (Rectified Flow-style)
  xt = t * x1 + (1 - t) * x0
  ut = x1 - x0               # constant velocity along the path

  # Predict velocity with the same UNet architecture
  v_pred = model(xt, t, global_cond=global_cond)

  # MSE loss
  L_FM = MSE(v_pred, ut)
  ```

- Integration with masks:
  - If an action mask is used (e.g., due to inpainting or varying horizons), we apply the *same* loss mask as DDPM so that only unmasked dimensions contribute to the loss.

This design keeps the **network, conditioning, and normalization identical** to DDPM; only the supervision signal and forward process differ.

### 3.4 Flow Matching Sampler Implementation

We replace the DDPM reverse sampler with a simple Euler ODE integrator over the learned velocity field:

```python
def flow_matching_sample(model, global_cond, action_shape, num_steps=4, device='cuda'):
    """
    Generate an action sequence using Euler ODE integration.

    Args:
        model: Trained ConditionalUnet1D
        global_cond: Encoded observation [B, cond_dim]
        action_shape: (B, T, Da)
        num_steps: Number of Euler steps (1, 2, 4, 8)
        device: Compute device

    Returns:
        x: Generated (normalized) action sequence
    """
    B, T, Da = action_shape

    # Start from pure noise (same prior as DDPM)
    x = torch.randn(B, T, Da, device=device)

    dt = 1.0 / num_steps
    for step in range(num_steps):
        t = torch.full((B,), step * dt, device=device)
        v = model(x, t, global_cond=global_cond)
        x = x + v * dt

    return x
```

The policy wrapper is responsible for:
- Passing the same `global_cond` as in DDPM.  
- Unnormalizing the final actions and selecting the correct horizon window (`n_obs_steps`, `n_action_steps`), just as in `DiffusionUnetImagePolicy.predict_action`.

---

## 4. Project Structure

```
Diffusion-Policy-Flow-Matching/
├── README.md                           # Project overview and setup
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Git ignore file
│
├── docs/
│   └── master_plan.md                  # This file - development bible
│
├── diffusion_policy/                   # Cloned official repo (as submodule)
│   └── ...                             # Original DP codebase
│
├── dpfm/                               # Our Flow Matching extensions
│   ├── __init__.py
│   ├── loss/
│   │   ├── __init__.py
│   │   └── flow_matching_loss.py       # FM loss implementation
│   │
│   ├── sampler/
│   │   ├── __init__.py
│   │   └── euler_sampler.py            # Euler ODE sampler (1,2,4,8 steps)
│   │
│   ├── policy/
│   │   ├── __init__.py
│   │   └── flow_matching_unet_image_policy.py  # FM policy wrapper
│   │
│   ├── workspace/
│   │   ├── __init__.py
│   │   └── train_fm_unet_image_workspace.py    # FM training workspace
│   │
│   └── utils/
│       ├── __init__.py
│       └── metrics.py                  # Latency and jerk logging
│
├── configs/
│   ├── baseline_ddpm_pusht.yaml        # Baseline DDPM config for Push-T
│   ├── fm_pusht_1step.yaml             # FM config with 1 step
│   ├── fm_pusht_2step.yaml             # FM config with 2 steps
│   └── fm_pusht_4step.yaml             # FM config with 4 steps
│
├── scripts/
│   ├── train_baseline.sh               # Train DDPM baseline
│   ├── train_fm.sh                     # Train Flow Matching
│   ├── eval_all.sh                     # Evaluate all models
│   └── generate_plots.py               # Generate result figures
│
├── data/                               # Data directory (gitignored)
│   ├── pusht/                          # Push-T dataset
│   └── robomimic/                      # Robomimic datasets
│
├── outputs/                            # Training outputs (gitignored)
│   └── ...
│
└── results/                            # Final results for report
    ├── figures/
    └── tables/
```

---

## 5. Implementation Plan

### Phase 1: Core Implementation (Day 1-2)

#### 5.1.1 Create Flow Matching Loss Module

**File**: `dpfm/loss/flow_matching_loss.py`

```python
"""
Flow Matching / Rectified Flow-style Loss for Diffusion Policy.

Key differences from DDPM:
1. Time t is continuous in [0, 1] instead of discrete timesteps
2. We predict velocity (u_t = x_1 - x_0) instead of noise
3. Interpolation is linear: x_t = t*x_1 + (1-t)*x_0 (linear OT path)
"""

import torch
import torch.nn.functional as F

class FlowMatchingLoss:
    """
    Conditional Flow Matching / Rectified Flow-style loss for action sequences.

    Uses a linear optimal-transport path: straight line from Gaussian noise to data.
    """
    
    def __init__(self, sigma_min=0.001):
        """
        Args:
            sigma_min: Minimum noise scale to avoid numerical issues at t=1
        """
        self.sigma_min = sigma_min
    
    def __call__(self, model, x_1, global_cond=None):
        """
        Compute flow matching loss.
        
        Args:
            model: Network that predicts velocity v_θ(x_t, t, cond)
            x_1: Target data (action sequence) [B, T, D]
            global_cond: Conditioning features [B, cond_dim]
        
        Returns:
            loss: Scalar MSE loss
        """
        batch_size = x_1.shape[0]
        device = x_1.device
        
        # Sample t uniformly
        t = torch.rand(batch_size, device=device)
        
        # Sample x_0 from standard Gaussian
        x_0 = torch.randn_like(x_1)
        
        # Linear optimal transport interpolation
        t_expand = t.view(-1, 1, 1)
        x_t = t_expand * x_1 + (1 - t_expand) * x_0
        
        # Add small noise at t=1 for stability
        x_t = x_t + self.sigma_min * torch.randn_like(x_t)
        
        # Target velocity (constant along OT path)
        u_t = x_1 - x_0
        
        # Predict velocity (same UNet as DDPM)
        v_pred = model(x_t, t, global_cond=global_cond)
        
        # MSE loss
        loss = F.mse_loss(v_pred, u_t)
        
        return loss
```

#### 5.1.2 Create Euler Sampler

**File**: `dpfm/sampler/euler_sampler.py`

```python
"""
Euler ODE Sampler for Flow Matching.

Integrates the learned velocity field from t=0 (noise) to t=1 (data).
"""

import torch
import time

class EulerSampler:
    """
    Euler method sampler for Flow Matching.
    """
    
    def __init__(self, num_steps=4):
        """
        Args:
            num_steps: Number of Euler integration steps (1, 2, 4, or 8)
        """
        self.num_steps = num_steps
        self.dt = 1.0 / num_steps
    
    @torch.no_grad()
    def sample(self, model, shape, global_cond=None, device='cuda'):
        """
        Generate samples via Euler integration.
        
        Args:
            model: Velocity network v_θ(x, t, cond)
            shape: Output shape (B, T, D)
            global_cond: Conditioning [B, cond_dim]
            device: Compute device
        
        Returns:
            x: Generated samples
            latency_ms: Inference time in milliseconds
        """
        start_time = time.time()
        
        # Initialize from Gaussian noise
        x = torch.randn(shape, device=device)
        
        # Euler integration from t=0 to t=1
        for step in range(self.num_steps):
            t = torch.full((shape[0],), step * self.dt, device=device)
            v = model(x, t, global_cond=global_cond)
            x = x + v * self.dt
        
        latency_ms = (time.time() - start_time) * 1000
        
        return x, latency_ms
```

#### 5.1.3 Create Flow Matching Policy

**File**: `dpfm/policy/flow_matching_unet_image_policy.py`

This will be a modified version of `DiffusionUnetImagePolicy` that preserves the **public interface** and **conditioning structure** but swaps out the generative objective and sampler:

1. **Architecture parity**
   - Reuse the same `ConditionalUnet1D` and `MultiImageObsEncoder` (same `down_dims`, kernel sizes, etc.) to ensure comparability with the original Diffusion Policy paper.
   - Keep `obs_as_global_cond=True` and the same `LowdimMaskGenerator` behavior so that the observation/action masking matches the DDPM baseline.

2. **Training (`compute_loss`)**
   - Normalize observations and actions using the same `LinearNormalizer` as DP.
   - Encode observations exactly as in `DiffusionUnetImagePolicy.compute_loss` to obtain `global_cond`.
   - Replace the DDPM forward-diffusion + noise prediction with `FlowMatchingLoss`:
     - Treat the **normalized actions** `nactions` as `x_1`.
     - Sample `x_0 ~ N(0, I)` and `t ~ U(0,1)`, build `x_t`, and call the shared UNet with `(x_t, t, global_cond)`.
     - Compute `MSE(v_pred, x_1 - x_0)`, optionally masked using the same `condition_mask`-derived loss mask as DDPM.

3. **Inference (`predict_action`)**
   - Keep the same `obs_dict` → `global_cond` pipeline and horizon slicing (`n_obs_steps`, `n_action_steps`).
   - Replace `conditional_sample` + DDPM scheduler with a call to `EulerSampler.sample`, starting from Gaussian noise in action space with the same shape and using the same `global_cond`.
   - Unnormalize actions using `self.normalizer['action'].unnormalize`, so that the outputs live in the same control space as the baseline policy.

4. **Latency logging hook**
   - Wrap sampling in a timer and return `latency_ms` along with actions so that the workspace/env_runner can log percentile latency while sharing the same rollout code as DP.

5. **Fairness guarantee**
   - From the workspace and env side, both policies expose the same `predict_action` signature and action statistics (mean, std via the same normalizer). Only the loss and sampling internals differ.

### Phase 2: Training Integration (Day 2-3)

#### 5.2.1 Create Training Workspace

**File**: `dpfm/workspace/train_fm_unet_image_workspace.py`

Key modifications from `TrainDiffusionUnetImageWorkspace`:
- **Policy instantiation**
  - Change the policy target from `diffusion_policy.policy.diffusion_unet_image_policy.DiffusionUnetImagePolicy`
    to `dpfm.policy.flow_matching_unet_image_policy.FlowMatchingUnetImagePolicy`.
  - Keep all other `cfg.policy` fields (shape_meta, encoder config, UNet hyperparameters, etc.) identical.

- **Training loop**
  - Reuse the same training loop structure, optimizer, EMA, dataloaders, LR scheduler, and logging/rollout schedule.
  - The workspace continues to call `model.compute_loss(batch)` and `policy.predict_action(obs_dict)`; the FM policy implements these methods with the Flow Matching objective described in 5.1.3.

- **Latency & jerk logging (evaluation only)**
  - After each rollout, post-process the recorded **denormalized actions** to compute:
    - Per-step latency statistics (p50, p95) based on the `latency_ms` returned by the policy or measured externally.
    - Mean action jerk (see Section 6.5) from the unnormalized action trajectories.
  - Log these metrics alongside success rate so that DDPM and FM runs go through the same evaluation code paths.

#### 5.2.2 Create Config Files

**File**: `configs/fm_pusht_4step.yaml`

```yaml
defaults:
  - _self_
  - task: pusht_image

name: train_fm_unet_image
_target_: dpfm.workspace.train_fm_unet_image_workspace.TrainFMUnetImageWorkspace

policy:
  _target_: dpfm.policy.flow_matching_unet_image_policy.FlowMatchingUnetImagePolicy
  
  # Key difference: no noise_scheduler, use our sampler
  sampler:
    _target_: dpfm.sampler.euler_sampler.EulerSampler
    num_steps: 4  # Change this for 1, 2, 4, 8 step experiments
  
  # Same encoder and backbone
  obs_encoder:
    _target_: diffusion_policy.model.vision.multi_image_obs_encoder.MultiImageObsEncoder
    # ... same as baseline
  
  # UNet unchanged
  diffusion_step_embed_dim: 128
  down_dims: [512, 1024, 2048]

# Training unchanged
training:
  num_epochs: 8000
  # ...
```

**Fairness with respect to DP original config**:
- The FM configs are cloned from the official Push-T image config used by Diffusion Policy (dataset path, horizon, obs/action history length, optimizer, batch size, LR schedule, EMA, rollout/eval schedule).
- The only config differences between `baseline_ddpm_pusht.yaml` and `fm_pusht_*.yaml` are:
  - `policy._target_` (DDPM vs FM implementation)
  - `policy.sampler` block (DDPM scheduler vs EulerSampler with different `num_steps`)
  - The experiment `name` and `num_steps` metadata for logging/plotting.

### Phase 3: Baseline & Experiments (Day 3-5)

#### 5.3.1 Train Baseline

```bash
# Train DDPM baseline on Push-T
python diffusion_policy/train.py \
    --config-name=train_diffusion_unet_image_workspace \
    task=pusht_image \
    training.seed=42
```

#### 5.3.2 Train Flow Matching Variants

```bash
# Train FM with 4 steps
python dpfm/train.py \
    --config-name=fm_pusht_4step \
    training.seed=42

# Repeat for 1, 2, 8 steps
```

### Phase 4: Evaluation & Analysis (Day 5-6)

#### 5.4.1 Metrics Collection

For each model checkpoint:

```python
# Collect during evaluation
results = {
    'success_rate': [],
    'latency_p50': [],
    'latency_p95': [],
    'jerk_mean': [],
    'num_steps': [],
}
```

#### 5.4.2 Generate Figures

1. **Steps vs Success Rate** (Pareto curve)
2. **Steps vs Latency** (p50 and p95)
3. **Steps vs Jerk**
4. **Success vs Latency** (Pareto frontier)

---

## 6. Experiment Design

### 6.1 Primary Environment: Push-T

**Why Push-T**:
- Fast to train and evaluate
- Clear success metric (coverage percentage)
- Well-tested in original DP paper
- Image-based observation

### 6.2 Experiment Matrix

| Method | Steps | Seeds | Est. Train Time |
|--------|-------|-------|-----------------|
| DDPM Baseline | 100 | 3 | 8h per seed |
| FM-DP | 1 | 3 | 8h per seed |
| FM-DP | 2 | 3 | 8h per seed |
| FM-DP | 4 | 3 | 8h per seed |

### 6.3 Evaluation Protocol

For each trained model:
1. Load checkpoint
2. Run 50 evaluation episodes
3. Record: success, latency per step, action trajectory
4. Compute: mean success, p50/p95 latency, mean jerk

The evaluation script/workspace should:
- Use the **same env_runner config** and evaluation episode length as the official Diffusion Policy Push-T experiments.
- Reuse the same seeding and reset behavior so that DDPM and FM differ only in their policy internals, not in environment stochasticity handling.

### 6.4 Statistical Reporting

- Report mean ± std across seeds
- Use paired t-test for significance
- Show confidence intervals in plots

### 6.5 Action Jerk Metric Implementation

To make the jerk metric physically meaningful and directly comparable to the baseline:

- **Signal space**
  - All jerk computations are done on **denormalized actions** `a_t`, obtained via `self.normalizer['action'].unnormalize(naction_pred)` inside the policy, so that values correspond to the same control units as in the original Diffusion Policy.

- **Definition**
  - For an action sequence `{a_0, ..., a_{T-1}}` from a rollout, define:
    - `Δa_t = a_t - a_{t-1}` for `t = 1, ..., T-1`
    - `jerk_t = ||Δa_t||²`
  - The reported jerk metric is:
    - `jerk_mean = (1 / (T-1)) * Σ_{t=1}^{T-1} jerk_t` averaged over all episodes.

- **Implementation location**
  - Implemented in `dpfm/utils/metrics.py` and called from the FM workspace (and optionally the baseline workspace), so that **both DDPM and FM** use the same post-processing code on their rollout logs.

---

## 7. Timeline

| Day | Date | Tasks |
|-----|------|-------|
| 1 | Nov 26 | ✅ Setup repo, env, clone DP, create plan |
| 2 | Nov 27 | Implement FM loss & sampler, create FM policy |
| 3 | Nov 28 | Create workspace, configs, start baseline training |
| 4 | Nov 29 | Debug FM training, run all seeds |
| 5 | Nov 30 | Evaluation & latency collection |
| 6 | Dec 1 | Generate figures, write analysis |
| 7 | Dec 2 | Write report draft |
| 8 | Dec 3 | Record video presentation |
| 9 | Dec 4 | Final polish, buffer |
| 10 | Dec 5 | **Submission** |

---

## 8. Risk Assessment & Fallback

### 8.1 Risk: FM Training Instability

**Symptoms**: Loss doesn't converge, NaN values, poor generation

**Mitigations**:
1. Add gradient clipping
2. Use lower learning rate
3. Add small noise (`sigma_min`) at t=1
4. Use EMA model for evaluation

**Fallback**: If FM fails completely, compare DDPM with truncated steps (10, 20, 50) to still demonstrate the latency-quality tradeoff.

### 8.2 Risk: Environment Issues

**Mitigations**:
- Use Push-T first (pure Python, no mujoco issues)
- Have conda environment export for reproducibility

### 8.3 Risk: Time Overrun

**Contingency**:
- Reduce seeds to 2 instead of 3
- Focus only on Push-T, drop robomimic
- Use pre-trained baseline if available

---

## 9. Deliverables

### 9.1 Code (Part 5)

- [ ] GitHub repo with full README
- [ ] One-command setup: `conda env create -f environment.yaml`
- [ ] One-command training: `bash scripts/train_all.sh`
- [ ] One-command evaluation: `bash scripts/eval_all.sh`

### 9.2 Report (Part 3, 2-4 pages)

1. **Introduction** (0.5 page): DP pain point, FM solution
2. **Method** (1 page): FM loss, sampler, integration
3. **Experiments** (0.5 page): Setup, metrics
4. **Results** (1 page): Key figures, analysis
5. **Conclusion**: Summary, future work, video link

### 9.3 Video (Part 2, 7 minutes)

- 0:00-0:45: DP overview and pain point
- 0:45-2:00: FM modification and intuition
- 2:00-4:30: Results and demo clips
- 4:30-6:00: Ablations and analysis
- 6:00-7:00: Conclusion and future work

### 9.4 Figures Checklist

- [ ] Fig 1: Method diagram (DDPM vs FM)
- [ ] Fig 2: Steps vs Success Rate
- [ ] Fig 3: Steps vs Latency (p50, p95)
- [ ] Fig 4: Steps vs Action Jerk
- [ ] Table 1: Summary of all experiments

---

## 10. References

1. Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion", RSS 2023
2. Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023
3. Liu et al., "Rectified Flow: A Marginal Preserving Approach to Optimal Transport", arXiv 2023
4. Song et al., "Consistency Models", ICML 2023

---

## Appendix A: Quick Commands

```bash
# Activate environment
conda activate DPFM

# Download Push-T dataset
cd diffusion_policy
mkdir -p data && cd data
wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip
unzip pusht.zip && rm pusht.zip
cd ../..

# Train baseline DDPM
cd diffusion_policy
python train.py \
    --config-name=train_diffusion_unet_image_workspace \
    task=pusht_image \
    training.seed=42 \
    training.device=cuda:0 \
    logging.mode=offline \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_baseline_pusht'

# Train FM (after implementation)
cd ..
python -m dpfm.train \
    --config-path=configs \
    --config-name=fm_pusht_4step \
    training.seed=42

# Evaluate
python -m dpfm.eval --checkpoint=outputs/fm_pusht_4step/checkpoints/latest.ckpt
```

---

## Appendix B: Key File Locations in Original DP

| Purpose | File Path |
|---------|-----------|
| Main Policy | `diffusion_policy/policy/diffusion_unet_image_policy.py` |
| UNet Model | `diffusion_policy/model/diffusion/conditional_unet1d.py` |
| Image Encoder | `diffusion_policy/model/vision/multi_image_obs_encoder.py` |
| Training Loop | `diffusion_policy/workspace/train_diffusion_unet_image_workspace.py` |
| Push-T Dataset | `diffusion_policy/dataset/pusht_image_dataset.py` |
| Push-T Env | `diffusion_policy/env/pusht/pusht_image_env.py` |
| Configs | `diffusion_policy/config/` |

---

## Appendix C: Expected Results Format

### Success Rate Table

| Method | Steps | Success Rate | p95 Latency (ms) | Jerk |
|--------|-------|--------------|------------------|------|
| DDPM | 100 | 85.2 ± 2.1 | 95.3 ± 5.2 | 0.023 |
| FM-DP | 1 | ??? | ??? | ??? |
| FM-DP | 2 | ??? | ??? | ??? |
| FM-DP | 4 | ??? | ??? | ??? |

### Pareto Plot

```
Success Rate
    ^
    |    * DDPM-100
    |  * FM-4
    | * FM-2
    |* FM-1
    +----------------> Latency
```

---

*This document is the single source of truth for the DPFM project. All development decisions should reference this plan.*
