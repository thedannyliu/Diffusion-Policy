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

**DDPM (current)**:
```
Forward: x_t = √(α_t) * x_0 + √(1-α_t) * ε
Loss: L = ||ε_θ(x_t, t) - ε||²
Inference: x_{t-1} = f(x_t, ε_θ(x_t, t))  [100 steps]
```

**Flow Matching (proposed)**:
```
Interpolation: x_t = t * x_1 + (1-t) * x_0  [x_1 = data, x_0 = noise]
Target velocity: u_t = x_1 - x_0
Loss: L = ||v_θ(x_t, t) - u_t||²
Inference: x_1 = x_0 + Σ v_θ(x_t, t) * Δt  [1-4 steps]
```

---

## 3. Technical Approach

### 3.1 What We Keep Unchanged

| Component | File | Reason |
|-----------|------|--------|
| Visual Encoder | `multi_image_obs_encoder.py` | Isolate the experiment variable |
| UNet Backbone | `conditional_unet1d.py` | Same architecture, different objective |
| Data Pipeline | `robomimic_replay_image_dataset.py` | Same data for fair comparison |
| Normalizer | `normalizer.py` | Same preprocessing |

### 3.2 What We Modify

| Component | Original | Modified |
|-----------|----------|----------|
| Training Loss | DDPM noise prediction | Flow Matching velocity prediction |
| Sampling | DDPM reverse diffusion (100 steps) | Euler ODE integration (1-4 steps) |
| Time Embedding | Discrete timesteps | Continuous t ∈ [0, 1] |

### 3.3 Flow Matching Loss Implementation

```python
def flow_matching_loss(model, obs, action, obs_encoder):
    """
    Compute Flow Matching loss for action diffusion.
    
    Args:
        model: ConditionalUnet1D that outputs velocity
        obs: Observation dict with images
        action: Ground truth action sequence [B, T, Da]
        obs_encoder: Visual encoder for conditioning
    
    Returns:
        loss: MSE loss between predicted and target velocity
    """
    batch_size = action.shape[0]
    device = action.device
    
    # 1. Sample random time t ~ U(0, 1)
    t = torch.rand(batch_size, device=device)
    
    # 2. Sample noise x_0 ~ N(0, I)
    x_0 = torch.randn_like(action)
    
    # 3. Interpolate: x_t = t * action + (1-t) * x_0
    #    Note: t=1 is data, t=0 is noise
    t_expand = t.view(batch_size, 1, 1)
    x_t = t_expand * action + (1 - t_expand) * x_0
    
    # 4. Target velocity: u_t = action - x_0 (constant along path)
    u_t = action - x_0
    
    # 5. Encode observations
    obs_features = obs_encoder(obs)
    global_cond = obs_features.reshape(batch_size, -1)
    
    # 6. Predict velocity: v_θ(x_t, t, obs)
    v_pred = model(x_t, t, global_cond=global_cond)
    
    # 7. MSE Loss
    loss = F.mse_loss(v_pred, u_t)
    
    return loss
```

### 3.4 Flow Matching Sampler Implementation

```python
def flow_matching_sample(model, obs_features, action_shape, num_steps=4, device='cuda'):
    """
    Generate action sequence using Euler ODE integration.
    
    Args:
        model: Trained ConditionalUnet1D
        obs_features: Encoded observation [B, obs_dim]
        action_shape: (B, T, Da)
        num_steps: Number of Euler steps (1, 2, 4, 8)
        device: Compute device
    
    Returns:
        action: Generated action sequence
    """
    batch_size, horizon, action_dim = action_shape
    
    # Start from pure noise
    x = torch.randn(batch_size, horizon, action_dim, device=device)
    
    # Time steps: from t=0 (noise) to t=1 (data)
    dt = 1.0 / num_steps
    
    for step in range(num_steps):
        t = torch.full((batch_size,), step * dt, device=device)
        
        # Predict velocity
        v = model(x, t, global_cond=obs_features)
        
        # Euler step: x_{t+dt} = x_t + v * dt
        x = x + v * dt
    
    return x
```

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
Flow Matching Loss for Diffusion Policy.

Key differences from DDPM:
1. Time t is continuous in [0, 1] instead of discrete timesteps
2. We predict velocity (u_t = x_1 - x_0) instead of noise
3. Interpolation is linear: x_t = t*x_1 + (1-t)*x_0
"""

import torch
import torch.nn.functional as F

class FlowMatchingLoss:
    """
    Conditional Flow Matching loss for action sequences.
    
    Uses optimal transport path: straight line from noise to data.
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
        
        # Optimal transport interpolation
        t_expand = t.view(-1, 1, 1)
        x_t = t_expand * x_1 + (1 - t_expand) * x_0
        
        # Add small noise at t=1 for stability
        x_t = x_t + self.sigma_min * torch.randn_like(x_t)
        
        # Target velocity (constant along OT path)
        u_t = x_1 - x_0
        
        # Predict velocity
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

This will be a modified version of `DiffusionUnetImagePolicy` that:
1. Uses `FlowMatchingLoss` instead of DDPM noise loss
2. Uses `EulerSampler` instead of DDPM reverse sampling
3. Logs latency and jerk metrics

### Phase 2: Training Integration (Day 2-3)

#### 5.2.1 Create Training Workspace

**File**: `dpfm/workspace/train_fm_unet_image_workspace.py`

Key modifications from `TrainDiffusionUnetImageWorkspace`:
- Replace `compute_loss` with flow matching loss
- Add latency logging during validation
- Add jerk computation for action sequences

#### 5.2.2 Create Config Files

**File**: `configs/fm_pusht_4step.yaml`

```yaml
defaults:
  - _self_
  - task: pusht_image

name: train_fm_unet_image
_target_: dpfm.workspace.train_fm_unet_image_workspace.TrainFMUnetImageWorkspace

# Same architecture as baseline
horizon: 16
n_obs_steps: 2
n_action_steps: 8

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

### 6.4 Statistical Reporting

- Report mean ± std across seeds
- Use paired t-test for significance
- Show confidence intervals in plots

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
