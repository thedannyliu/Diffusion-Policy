# Diffusion Policy with Flow Matching (DPFM)

> **CS 7643 Deep RL Final Project**: Extending Diffusion Policy with Flow Matching for Faster Inference

[![Python 3.9](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red.svg)](https://pytorch.org/)

## 🎯 Project Overview

This project extends the [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) paper by replacing the DDPM sampling objective with **Flow Matching**, enabling action generation with significantly fewer sampling steps (1-4 steps vs 100 steps) while maintaining comparable task success rates.

### Research Question

> Can Flow Matching training enable Diffusion Policy to achieve comparable success rates with 1-4 sampling steps, while significantly reducing control latency and action jerk?

## 📊 Key Results

| Method | Steps | Success Rate | p95 Latency | Action Jerk |
|--------|-------|--------------|-------------|-------------|
| DDPM (Baseline) | 100 | TBD | TBD | TBD |
| FM-DP | 4 | TBD | TBD | TBD |
| FM-DP | 2 | TBD | TBD | TBD |
| FM-DP | 1 | TBD | TBD | TBD |

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone this repository
git clone git@github.com:thedannyliu/Diffusion-Policy-Flow-Matching.git
cd Diffusion-Policy-Flow-Matching

# Create conda environment
conda create -n DPFM python=3.9 -y
conda activate DPFM

# Install PyTorch (CUDA 11.8)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install einops diffusers hydra-core omegaconf h5py zarr numcodecs \
    matplotlib wandb tensorboard tqdm dill scikit-image imageio \
    imageio-ffmpeg termcolor psutil click gymnasium pymunk pygame shapely av
```

### 2. Download Data

```bash
cd diffusion_policy
mkdir -p data && cd data
wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip
unzip pusht.zip && rm pusht.zip
cd ../..
```

### 3. Train Baseline (DDPM)

```bash
cd diffusion_policy
python train.py \
    --config-name=train_diffusion_unet_image_workspace \
    task=pusht_image \
    training.seed=42 \
    training.device=cuda:0 \
    logging.mode=offline
```

### 4. Train Flow Matching (After Implementation)

```bash
python -m dpfm.train \
    --config-path=configs \
    --config-name=fm_pusht_4step \
    training.seed=42
```

## 📁 Project Structure

```
Diffusion-Policy-Flow-Matching/
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── .gitignore
│
├── docs/
│   └── master_plan.md           # Detailed development plan
│
├── diffusion_policy/            # Official DP codebase (cloned)
│   ├── diffusion_policy/        # Core modules
│   │   ├── policy/              # Policy implementations
│   │   ├── model/               # Neural network models
│   │   ├── dataset/             # Dataset loaders
│   │   ├── workspace/           # Training workspaces
│   │   └── config/              # Hydra configs
│   └── data/                    # Datasets (gitignored)
│
├── dpfm/                        # Our Flow Matching extension
│   ├── loss/
│   │   └── flow_matching_loss.py
│   ├── sampler/
│   │   └── euler_sampler.py
│   ├── policy/
│   │   └── flow_matching_unet_image_policy.py
│   └── workspace/
│       └── train_fm_unet_image_workspace.py
│
├── configs/                     # Experiment configs
│   ├── fm_pusht_1step.yaml
│   ├── fm_pusht_2step.yaml
│   └── fm_pusht_4step.yaml
│
├── scripts/                     # Training/eval scripts
│   ├── train_baseline.sh
│   ├── train_fm.sh
│   └── generate_plots.py
│
└── results/                     # Figures for report
    └── figures/
```

## 🔬 Method

### Flow Matching vs DDPM

**DDPM (Original Diffusion Policy)**:
- Forward: Gradually add noise over T timesteps
- Training: Predict noise ε from noisy sample
- Inference: Iterative denoising (100 steps)

**Flow Matching (Our Extension)**:
- Forward: Linear interpolation between noise and data
- Training: Predict velocity field v
- Inference: Single ODE integration (1-4 Euler steps)

### Key Equations

**Flow Matching Loss**:
```
x_t = t * x_1 + (1-t) * x_0    (interpolation)
u_t = x_1 - x_0                (target velocity)
L = ||v_θ(x_t, t, obs) - u_t||²
```

**Euler Sampling**:
```
x_{k+1} = x_k + v_θ(x_k, t_k) * Δt
```

## 📈 Experiments

### Environment: Push-T

The Push-T task requires a robot to push a T-shaped block into a target zone. We use image-based observations (96×96 RGB).

### Metrics

1. **Success Rate**: Task completion percentage
2. **Inference Latency**: p50/p95 per control step (ms)
3. **Action Jerk**: Mean ||a_t - a_{t-1}||²

### Experiment Matrix

| Configuration | Description |
|---------------|-------------|
| DDPM-100 | Baseline with 100 denoising steps |
| FM-1 | Flow Matching with 1 Euler step |
| FM-2 | Flow Matching with 2 Euler steps |
| FM-4 | Flow Matching with 4 Euler steps |

## 📝 Report & Video

- **Report**: `docs/final_report.pdf` (2-4 pages)
- **Video**: [YouTube Link - TBD]

## 👥 Team

- Team Member 1
- Team Member 2

## 📚 References

1. Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion", RSS 2023
2. Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023
3. Liu et al., "Rectified Flow", arXiv 2023

## 📄 License

This project is for educational purposes (CS 7643 Final Project). The original Diffusion Policy codebase is under MIT License.

## 🙏 Acknowledgments

- [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) authors
- CS 7643 Deep Reinforcement Learning course staff
