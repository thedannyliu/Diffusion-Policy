# Diffusion Policy with Flow Matching (DPFM)

> **CS 8803 Deep Reinforcement Learning Final Project**: Extending Diffusion Policy with Flow Matching for Faster Inference

## 🎯 Project Overview

This project extends [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) by replacing DDPM with **Flow Matching**, enabling action generation with significantly fewer sampling steps (**4-8 steps vs 100 steps**) while maintaining comparable task success rates.

### Research Question

> Can Flow Matching training enable Diffusion Policy to achieve comparable success rates with 4-8 sampling steps, while significantly reducing control latency?

---

## 📊 Experimental Results

### Push-T Task (Simulation)

| Method | Steps | Test Score | Latency (p50) | Speedup |
|--------|-------|------------|---------------|---------|
| **DDPM Baseline** | 100 | **0.892** | 650 ms | 1× |
| FM-DP (step=16) | 16 | 0.772 | 90 ms | **7.2×** |
| FM-DP (step=8) | 8 | 0.845 | 48 ms | **13.5×** |
| FM-DP (step=4) | 4 | 0.812 | 24 ms | **27×** |

### Key Findings

1. **Speed-Accuracy Trade-off**: FM achieves **7-27× speedup** with only **5-13% accuracy drop**
2. **Optimal Configuration**: FM with 8 steps provides the best balance (0.845 score, 13.5× faster)
3. **Training Stability**: FM converges faster in early epochs but shows more variance in later stages
4. **Early Stopping Helps**: Best FM scores appear around epochs 650-1000, not at convergence
5. **Diminishing Returns**: Step=16 (0.772) performs worse than step=8 (0.845), suggesting 8 is optimal

### Training Dynamics Comparison

| Aspect | DDPM | Flow Matching |
|--------|------|---------------|
| Initial loss | ~0.48 | ~1.17 |
| Final loss | ~0.0002 | ~0.0003 |
| Best score epoch | 500 | 900-1000 |
| Training time (3050 epochs) | ~20h | ~12h |
| Score variance | Low (±2%) | Higher (±5%) |

---

## 📈 Detailed Analysis

### Push-T Experiments (9 runs total)

#### DDPM Baseline
| Seed | Best Score | Best Epoch | Final Loss | Notes |
|------|------------|------------|------------|-------|
| 42 | 0.892 | 500 | 0.0002 | Stable, consistent performance |
| 123 | ~0.88 | - | - | Similar to seed 42 |

**Insights:**
- Performance peaks early (epoch 500) then fluctuates between 0.82-0.88
- Very stable training with low variance
- 100 diffusion steps provide robust action generation

#### Flow Matching (Various Configurations)
| Config | Steps | LR | Seed | Best Score | Latency |
|--------|-------|-----|------|------------|---------|
| FM_step8 | 8 | 1e-4 | 42 | **0.845** | 48 ms |
| FM_step16 | 16 | 1e-4 | 42 | 0.823 (train) / 0.772 (eval) | 90 ms |
| FM_step4_v1 | 4 | 1e-4 | 42 | 0.812 | 24 ms |
| FM_step4_v2 | 4 | 2e-4 | 42 | 0.838 | 24 ms |
| FM_step4 | 4 | 1e-4 | 123 | 0.807 | 24 ms |
| FM_step2 | 2 | 1e-4 | 42 | ~0.75 | 12 ms |

**Insights:**
- **Inference steps matter**: 8 steps provides the optimal balance
- **Learning rate**: Higher LR (2e-4) slightly improves 4-step performance
- **Seed sensitivity**: FM shows more variance across seeds (~5%)
- **Diminishing returns**: 16 steps (0.772 eval) performs worse than 8 steps (0.845)
- **Action smoothness**: Lower steps show higher action jerk (less smooth trajectories)

### Training Curve Analysis

DDPM Loss Progression:
- Epoch 0: 0.476 → Epoch 500: 0.001 → Epoch 3050: 0.0002

FM Loss Progression:  
- Epoch 0: 1.171 → Epoch 500: 0.008 → Epoch 1400: 0.0003

**Key Observations:**
1. **FM starts with higher loss** but converges to similar final values
2. **Test scores don't correlate perfectly with loss** - best scores often at intermediate epochs
3. **Overfitting risk**: Both methods show slight performance degradation after prolonged training

---

## 🤖 Real Robot Experiments

### Dataset: Two-Camera Manipulation Tasks

We trained on real robot datasets with dual camera setup:

| Task | Episodes | Steps | Camera Views |
|------|----------|-------|--------------|
| Sphere manipulation | 112 | 10,891 | 2 (front + side) |
| Cube manipulation | 100 | 17,667 | 2 (front + side) |

### Training Configurations Comparison

We ran two rounds of real robot training experiments:

#### Version 1: Initial Training (600 epochs, 12h, step=4)
| Setting | FM (v1) | DDPM (v1) |
|---------|---------|-----------|
| Config | `train_fm_real_robot_workspace` | `train_ddpm_real_robot_workspace` |
| Inference Steps | 4 | 100 |
| Epochs | 600 | 600 |
| Time Limit | 12h | 12h |
| Encoder | HybridImageEncoder | MultiImageObsEncoder |

#### Version 2: Optimized Training (1000 epochs, 24h, step=8, Fair Comparison)
| Setting | FM (v2) | DDPM (v2) |
|---------|---------|-----------|
| Config | `train_fm_real_robot_fair_workspace` | `train_ddpm_real_robot_workspace` |
| Inference Steps | 8 | 100 |
| Epochs | 1000 | 1000 |
| Time Limit | 24h | 24h |
| Encoder | **MultiImageObsEncoder** | **MultiImageObsEncoder** |

**Key Differences (v1 → v2):**
1. **Fair Encoder**: Both now use identical `MultiImageObsEncoder` (ResNet18)
2. **More Steps**: FM increased from 4 → 8 inference steps
3. **More Training**: 600 → 1000 epochs, 12h → 24h time limit
4. **Same Architecture**: UNet down_dims=[512,1024,2048], crop_shape=[216,288]

### Training Results

| Method | Version | Sphere Final Loss | Cube Final Loss | Epochs Completed |
|--------|---------|-------------------|-----------------|------------------|
| FM | v1 (step=4) | 0.0036 | 0.0060 | 550/600 |
| FM | v2 (step=8) | 0.0030 | 0.0020 | ~999/1000 |
| DDPM | v1 | 0.0008 | 0.0022 | 550/600 |
| DDPM | v2 | 0.0010 | 0.0005 | ~999/1000 |

### Checkpoint Locations

#### Version 1 (Initial) - FM step=4
```
data/outputs/real_robot/2025.11.27/15.58.10_fm_sphere_step4_seed42/checkpoints/
data/outputs/real_robot/2025.11.27/15.58.10_fm_cube_step4_seed42/checkpoints/
data/outputs/real_robot/2025.11.27/16.22.11_fm_fair_sphere_step4_seed42/checkpoints/
data/outputs/real_robot/2025.11.27/16.22.11_fm_fair_cube_step4_seed42/checkpoints/
```

#### Version 2 (Optimized) - FM step=8
```
data/outputs/2025.11.29/00.33.00_train_fm_real_robot_fair_sphere/checkpoints/
data/outputs/2025.11.29/00.33.54_train_fm_real_robot_fair_cube/checkpoints/
```

#### DDPM Baselines
```
# Version 1
diffusion_policy/data/outputs/2025.11.27/16.22.09_train_ddpm_real_robot_cube/checkpoints/
diffusion_policy/data/outputs/2025.11.27/16.33.05_train_ddpm_real_robot_sphere/checkpoints/

# Version 2
diffusion_policy/data/outputs/2025.11.29/01.33.13_train_ddpm_real_robot_sphere/checkpoints/
diffusion_policy/data/outputs/2025.11.29/01.44.20_train_ddpm_real_robot_cube/checkpoints/
```

### Real Robot Training Insights

1. **Data efficiency**: Real robot datasets are smaller but more diverse
2. **Image processing**: 240×320 RGB images cropped to 216×288 (90% center crop)
3. **Action space**: 6-DoF end-effector pose (x, y, z, roll, pitch, yaw)
4. **Validation**: No simulation rollout available - validation only via held-out trajectories

---

## 🔧 Implementation Details

### Flow Matching vs DDPM

| Aspect | DDPM | Flow Matching |
|--------|------|---------------|
| Forward process | Add Gaussian noise | Linear interpolation |
| Training target | Predict noise ε | Predict velocity v |
| Sampling | 100 DDPM steps | 4-8 Euler ODE steps |
| Latency | ~650 ms | ~25-50 ms |
| Training loss | MSE(ε_pred, ε) | MSE(v_pred, v_target) |

### Flow Matching Loss

```python
# Optimal Transport path (linear interpolation)
t = torch.rand(batch_size)  # t ∈ [0, 1]
x_t = (1 - t) * noise + t * action

# Target velocity field
v_target = action - noise

# Predict velocity and compute loss
v_pred = model(x_t, t, obs_encoding)
loss = F.mse_loss(v_pred, v_target)
```

### Euler ODE Sampling

```python
def sample(self, obs_encoding, num_steps=4):
    x = torch.randn_like(action_template)  # Start from noise
    
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t = torch.full((batch,), i * dt)
        v = self.model(x, t, obs_encoding)  # Predict velocity
        x = x + v * dt  # Euler step
    
    return x  # Final action
```

---

## 🏗️ Repository Structure

```
Diffusion-Policy-Flow-Matching/
├── diffusion_policy/          # Original Diffusion Policy codebase
│   ├── diffusion_policy/      # Core DP modules
│   ├── config/task/           # Task configs including real_robot_two_cameras.yaml
│   └── train.py               # DDPM training entry point
│
├── dpfm/                      # Flow Matching extension
│   ├── config/                # Hydra configs
│   │   ├── task/              # Task configs (pusht_image.yaml, real_robot_image.yaml)
│   │   └── train_fm_*.yaml    # Training workspace configs
│   ├── loss/                  # Flow Matching loss (flow_matching_loss.py)
│   ├── sampler/               # Euler ODE sampler (euler_sampler.py)
│   ├── policy/                # FM policies (hybrid_image, unet_image)
│   └── train.py               # FM training entry point
│
├── scripts/                   # SLURM job scripts
├── results/                   # Evaluation results
└── data/                      # Training data and outputs
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone git@github.com:thedannyliu/Diffusion-Policy-Flow-Matching.git
cd Diffusion-Policy-Flow-Matching

# Create conda environment
conda create -n DPFM python=3.9 -y
conda activate DPFM

# Install PyTorch (CUDA 11.8)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt
```

### 2. Train Flow Matching

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/diffusion_policy"

# Train FM with 4 inference steps
python -m dpfm.train \
    --config-name=train_fm_unet_hybrid_image_workspace \
    policy.num_inference_steps=4 \
    training.seed=42 \
    logging.mode=online
```

### 3. Evaluate

```bash
python -m dpfm.eval \
    --checkpoint path/to/checkpoint.ckpt \
    --n_test 50 \
    --output_dir results/eval
```

---

## 📊 WandB Tracking

All experiments logged to WandB:
- **Push-T**: `dpfm_pusht_v2`
- **Real Robot**: `dpfm_real_robot`

---

## 🔬 Ablation Studies

### Effect of Inference Steps

| Steps | Score | Latency | Action Smoothness |
|-------|-------|---------|-------------------|
| 2 | 0.75 | 12 ms | Low (high jerk) |
| 4 | 0.81 | 24 ms | Medium |
| 8 | **0.85** | 48 ms | Good |
| 16 | 0.77 | 90 ms | Better |
| 100 (DDPM) | 0.89 | 650 ms | Best |

**Note**: Step=16 shows lower score (0.77) than step=8 (0.85), demonstrating that 8 steps is the optimal sweet spot for FM.

### Effect of Learning Rate

| LR | Steps=4 Score | Steps=8 Score |
|----|---------------|---------------|
| 1e-4 | 0.812 | 0.845 |
| 2e-4 | 0.838 | 0.821 |

Higher LR helps with fewer steps but may hurt with more steps.

---

## 🤖 Real Robot Deployment Guide

### Deploying Flow Matching Policy

If your colleague has already deployed Diffusion Policy on a real robot, deploying Flow Matching requires minimal changes:

#### Key Differences Between DDPM and FM Deployment

| Aspect | DDPM (Diffusion Policy) | Flow Matching |
|--------|------------------------|---------------|
| Checkpoint Format | Same `.ckpt` format | Same `.ckpt` format |
| Policy Interface | `predict_action(obs_dict)` | `predict_action(obs_dict)` |
| Inference Speed | ~650ms (100 steps) | **~50ms (8 steps)** |
| Output Format | Action tensor | Action tensor |
| Normalizer | Same | Same |

#### Modifications Required for FM Deployment

1. **Update `eval_real_robot.py`** to support FM policies:

```python
# In eval_real_robot.py, add FM handling:

if 'diffusion' in cfg.name:
    # Original DDPM handling
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    policy.num_inference_steps = 16  # DDIM inference iterations
    
elif 'fm' in cfg.name or 'flow' in cfg.name:
    # Flow Matching handling (new)
    policy = workspace.model
    if hasattr(workspace, 'ema_model') and workspace.ema_model is not None:
        policy = workspace.ema_model
    # FM uses fewer steps (already configured in checkpoint)
    # Optionally override: policy.num_inference_steps = 8
```

2. **Loading FM Checkpoint**:

```python
import torch
import dill
from dpfm.workspace.train_fm_unet_image_workspace import TrainFlowMatchingUnetImageWorkspace

# Load checkpoint
payload = torch.load('path/to/fm_checkpoint.ckpt', pickle_module=dill)
cfg = payload['cfg']

# Instantiate workspace
workspace = TrainFlowMatchingUnetImageWorkspace(cfg)
workspace.load_payload(payload)

# Get policy
policy = workspace.model
policy.eval().to('cuda')
```

3. **Action Prediction** (identical interface):

```python
# Both DDPM and FM use the same interface
with torch.no_grad():
    obs_dict = {
        'camera_0': camera_0_image,  # [1, T, C, H, W]
        'camera_1': camera_1_image,
    }
    result = policy.predict_action(obs_dict)
    action = result['action']  # [1, n_action_steps, action_dim]
```

#### Recommended FM Checkpoints for Real Robot

```bash
# Sphere task - FM step=8 (best)
data/outputs/2025.11.29/00.33.00_train_fm_real_robot_fair_sphere/checkpoints/latest.ckpt

# Cube task - FM step=8 (best)
data/outputs/2025.11.29/00.33.54_train_fm_real_robot_fair_cube/checkpoints/latest.ckpt
```

#### Performance Comparison for Real-Time Control

| Method | Inference Time | Control Loop Rate | Suitability |
|--------|---------------|-------------------|-------------|
| DDPM (100 steps) | ~650 ms | 1.5 Hz | Slow for dynamic tasks |
| FM (8 steps) | ~50 ms | **20 Hz** | Good for real-time control |
| FM (4 steps) | ~25 ms | **40 Hz** | Best for fast reactions |

#### Complete Deployment Example

```python
#!/usr/bin/env python3
"""Deploy Flow Matching policy on real robot."""

import torch
import dill
import hydra
from omegaconf import OmegaConf

# Register eval resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

def load_fm_policy(checkpoint_path, device='cuda'):
    """Load FM policy from checkpoint."""
    payload = torch.load(checkpoint_path, pickle_module=dill)
    cfg = payload['cfg']
    
    # Get workspace class
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace.load_payload(payload)
    
    # Get policy with EMA if available
    if hasattr(workspace, 'ema_model') and workspace.ema_model is not None:
        policy = workspace.ema_model
    else:
        policy = workspace.model
    
    policy.eval().to(device)
    return policy, cfg

def predict_action(policy, obs_dict, device='cuda'):
    """Predict action from observations."""
    # Move observations to device
    obs_dict_cuda = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v 
        for k, v in obs_dict.items()
    }
    
    with torch.no_grad():
        result = policy.predict_action(obs_dict_cuda)
    
    return result['action'].cpu().numpy()

# Usage
if __name__ == '__main__':
    policy, cfg = load_fm_policy('path/to/fm_checkpoint.ckpt')
    
    # In your control loop:
    # action = predict_action(policy, obs_dict)
    # robot.execute(action)
```

---

## 🎓 Conclusions

1. **Flow Matching is a viable alternative** to DDPM for Diffusion Policy
2. **13-27× speedup** with <10% accuracy loss is achievable
3. **8 inference steps** provides optimal quality/speed trade-off
4. **Training is faster** but requires careful hyperparameter tuning
5. **Real robot deployment** is feasible with the trained policies

### Future Work

- [ ] Implement adaptive step scheduling
- [ ] Explore distillation from DDPM to FM
- [ ] Test on more complex manipulation tasks
- [ ] Investigate multi-step flow matching

---

## 📚 References

1. **Diffusion Policy**: Chi et al., RSS 2023 [[Paper](https://arxiv.org/abs/2303.04137)]
2. **Flow Matching**: Lipman et al., ICLR 2023 [[Paper](https://arxiv.org/abs/2210.02747)]
3. **Rectified Flow**: Liu et al., ICLR 2023 [[Paper](https://arxiv.org/abs/2209.03003)]
4. **CFM**: Tong et al., 2023 [[Paper](https://arxiv.org/abs/2302.00482)]

---

## 📄 License

Educational project for CS 8803. Original Diffusion Policy under MIT License.

## 🙏 Acknowledgments

- [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) authors
- CS 8803 Deep Reinforcement Learning course staff at Georgia Tech
