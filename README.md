# Flow Matching for Diffusion Policy: Faster Visuomotor Control

**CS 8803 Deep Reinforcement Learning - Final Project**

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![PyTorch 2.0](https://img.shields.io/badge/pytorch-2.0-red.svg)](https://pytorch.org/)

<p align="center">
  <img src="docs/media/fm_vs_ddpm.gif" alt="Flow Matching vs DDPM comparison" width="600">
</p>

## 🎯 Project Overview

This project extends [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) (Chi et al., RSS 2023) by replacing the DDPM (Denoising Diffusion Probabilistic Model) with **Flow Matching**, achieving:

- **26× faster inference** (24ms vs 650ms) 
- **97% of DDPM performance** (0.844 vs 0.869 on PushT)
- **3× faster training** (6-9 hours vs 19 hours)

### Key Insight

Flow Matching learns a direct velocity field $v_\theta(x_t, t)$ to transport noise $x_0 \sim \mathcal{N}(0, I)$ to actions $x_1$ along optimal transport paths, requiring only 4-16 Euler steps vs 100 DDPM steps.

---

## 📊 Main Results

| Method | Steps | Score | Latency (p50) | Training Time | Speedup |
|--------|-------|-------|---------------|---------------|---------|
| **DDPM Baseline** | 100 | **0.869** | 650.0 ms | 19.4 hr | 1.0× |
| FM (16 steps) | 16 | 0.844 | 150.7 ms | 8.9 hr | **4.3×** |
| FM (lr=5e-5) | 4 | 0.820 | 24.5 ms | 6.8 hr | **26.5×** |
| FM (8 steps) | 8 | 0.801 | 48.0 ms | 8.0 hr | **13.5×** |
| FM (4 steps) | 4 | 0.777 | 24.5 ms | 6.5 hr | **26.5×** |

**Best Speed-Quality Trade-off**: FM with 16 steps achieves 97% of DDPM performance with 4.3× speedup.

**Best Real-time Performance**: FM with 4 steps and lr=5e-5 achieves 94% of DDPM with 26.5× speedup.

For detailed ablation results, see [docs/results.md](docs/results.md).

---

## 🏗️ Repository Structure

```
Diffusion-Policy-Flow-Matching/
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── notebooks/
│   └── results_analysis.ipynb    # Jupyter notebook reproducing key results
├── dpfm/                         # Flow Matching implementation
│   ├── config/                   # Hydra configuration files
│   │   ├── train_fm_unet_hybrid_image_workspace.yaml
│   │   ├── train_ddpm_unet_hybrid_pusht.yaml
│   │   └── task/pusht_image.yaml
│   ├── loss/                     # Flow matching loss (CFM)
│   ├── sampler/                  # Euler ODE sampler
│   ├── policy/                   # FM policy wrapper
│   ├── train.py                  # Training entry point
│   └── eval.py                   # Evaluation entry point
├── diffusion_policy/             # Original Diffusion Policy (baseline)
├── scripts/                      # SLURM job scripts
│   ├── submit_experiments.sh     # Submit ablation experiments
│   └── eval.sh                   # Evaluation script
├── results/                      # Evaluation outputs
│   ├── eval_baseline/            # DDPM results + videos
│   ├── eval_fm_4step/            # FM 4-step results
│   └── eval_fm_step16/           # FM 16-step results
├── logs/                         # Training logs
│   └── experiments/              # SLURM job logs
└── docs/                         # Documentation
    ├── results.md                # Detailed experimental results
    └── ablation_design.md        # Ablation study design
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/thedannyliu/Diffusion-Policy-Flow-Matching.git
cd Diffusion-Policy-Flow-Matching

# Create conda environment
conda create -n DPFM python=3.9 -y
conda activate DPFM

# Install PyTorch (CUDA 11.8)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Install diffusion_policy in development mode
cd diffusion_policy && pip install -e . && cd ..
```

### 2. Download Data

#### PushT Dataset (Simulation)
```bash
mkdir -p data/training
cd data/training
wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip
unzip pusht.zip
cd ../..
```

#### Real Robot Data (Optional)
```bash
# Download from: [PLACEHOLDER - Add Google Drive/Dropbox link]
# Place in data/training/real_robot/
```

### 3. Training

```bash
# Set Python path
export PYTHONPATH="${PWD}:${PWD}/diffusion_policy:$PYTHONPATH"
cd diffusion_policy

# Train Flow Matching policy (4 inference steps, default lr=1e-4)
python dpfm/train.py --config-name=train_fm_unet_hybrid_image_workspace \
    policy.num_inference_steps=4 \
    training.seed=42

# Train Flow Matching with optimized learning rate
python dpfm/train.py --config-name=train_fm_unet_hybrid_image_workspace \
    policy.num_inference_steps=4 \
    optimizer.lr=5e-5 \
    training.seed=42

# Train DDPM baseline (100 inference steps)
python dpfm/train.py --config-name=train_ddpm_unet_hybrid_pusht \
    training.seed=42
```

### 4. Evaluation

```bash
# Evaluate a trained checkpoint
python dpfm/eval.py \
    --checkpoint_path data/outputs/2025.12.04/<run_dir>/checkpoints/latest.ckpt \
    --n_eval_episodes 50 \
    --output_dir results/eval_output
```

### 5. Reproduce Results

Open `notebooks/results_analysis.ipynb` in Jupyter to reproduce the key metrics and visualizations:

```bash
jupyter notebook notebooks/results_analysis.ipynb
```

---

## 🔬 Method

### Flow Matching vs DDPM

| Aspect | DDPM | Flow Matching |
|--------|------|---------------|
| Forward process | Gaussian noise schedule | Linear interpolation |
| Training target | Predict noise $\epsilon$ | Predict velocity $v$ |
| Sampling | 100 DDPM steps | 4-16 Euler steps |
| Inference latency | ~650 ms | ~25-150 ms |

### Flow Matching Training

```python
# Optimal Transport path (linear interpolation)
t = torch.rand(batch_size, 1, 1)  # t ∈ [0, 1]
x_t = (1 - t) * x_0 + t * x_1     # x_0 = noise, x_1 = action

# Target: velocity from noise to action
v_target = x_1 - x_0

# Predict velocity and compute loss
v_pred = model(x_t, t, obs_encoding)
loss = F.mse_loss(v_pred, v_target)
```

### Euler ODE Sampling

```python
def sample(self, obs_encoding, num_steps=4):
    x = torch.randn_like(action_template)  # Start from noise x_0
    dt = 1.0 / num_steps
    
    for i in range(num_steps):
        t = torch.full((batch,), i * dt)
        v = self.model(x, t, obs_encoding)  # Predict velocity
        x = x + v * dt                       # Euler integration
    
    return x  # Final action x_1
```

---

## 📈 Ablation Studies

### Effect of Inference Steps

| Steps | Score | Latency | Speedup |
|-------|-------|---------|---------|
| 4 | 0.777 | 24.5 ms | 26.5× |
| 8 | 0.801 | 48.0 ms | 13.5× |
| **16** | **0.844** | 150.7 ms | **4.3×** |
| 100 (DDPM) | 0.869 | 650.0 ms | 1× |

**Insight**: 16 steps achieves near-DDPM quality; 4-8 steps for real-time applications.

### Learning Rate Sensitivity

| Learning Rate | Score | Notes |
|---------------|-------|-------|
| 1e-4 (default) | 0.777 | Standard DDPM setting |
| **5e-5** | **0.820** | +5.5% improvement, more stable |
| 1e-3 | 0.741 | Too aggressive, lower score |

**Insight**: FM benefits from lower learning rate (5e-5) for stable training.

### Architecture Comparison

| Architecture | Score | Status |
|--------------|-------|--------|
| **UNet Hybrid** | 0.777+ | ✅ Works well |
| Transformer | 0.117 | ❌ Failed |

**Insight**: UNet architecture is essential for FM on PushT task.

---

## 📁 Data Format

### PushT Dataset Structure
```
data/training/pusht/
├── pusht_cchi_v7_replay.zarr/
│   ├── data/
│   │   ├── action/           # (N, 2) - delta x, delta y
│   │   ├── img/              # (N, 96, 96, 3) - RGB images
│   │   ├── keypoint/         # (N, 9, 2) - keypoints
│   │   └── state/            # (N, 5) - agent state
│   └── meta/
│       └── episode_ends      # Episode boundaries
```

### Real Robot Data Structure
```
data/training/real_robot/
├── two_cameras_cube/         # Cube manipulation task
│   └── *.hdf5
└── two_cameras_sphere/       # Sphere manipulation task
    └── *.hdf5
```

---

## 📚 References

1. Chi, C., et al. "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion." RSS 2023. [[Paper](https://arxiv.org/abs/2303.04137)] [[Code](https://github.com/real-stanford/diffusion_policy)]

2. Lipman, Y., et al. "Flow Matching for Generative Modeling." ICLR 2023. [[Paper](https://arxiv.org/abs/2210.02747)]

3. Liu, X., et al. "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow." ICLR 2023. [[Paper](https://arxiv.org/abs/2209.03003)]

---

## 📜 License

This project is for educational purposes (CS 8803 Deep Reinforcement Learning, Georgia Tech). The original Diffusion Policy code is under MIT License.

## 🙏 Acknowledgments

- [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) authors
- CS 8803 Deep Reinforcement Learning course staff at Georgia Tech
- Georgia Tech PACE cluster resources
