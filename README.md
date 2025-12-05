# Diffusion Policy with Flow Matching

**CS 8803 Deep Reinforcement Learning - Final Project**

## Project Overview

This project extends [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) (Chi et al., RSS 2023) by replacing the DDPM (Denoising Diffusion Probabilistic Model) sampling with **Flow Matching**, enabling significantly faster action generation with **4-8 inference steps vs. 100 steps** while maintaining comparable task success rates.

### Research Question

> Can Flow Matching training enable Diffusion Policy to achieve comparable success rates with 4-8 sampling steps, while significantly reducing control latency?

### Key Results

| Method | Inference Steps | Test Score | Latency (p50) | Speedup |
|--------|-----------------|------------|---------------|---------|
| **DDPM Baseline** | 100 | **0.869** | 650 ms | 1× |
| FM (lr=5e-5) | 4 | **0.816** | 24 ms | **27×** |
| FM (step=8) | 8 | 0.779 | 48 ms | **13.5×** |
| FM (step=16) | 16 | 0.757 | 151 ms | **4.3×** |
| FM (step=4) | 4 | 0.757 | 24 ms | **27×** |

**Key Finding**: With tuned learning rate (5e-5), Flow Matching achieves **0.816 score** (94% of DDPM) with **27× speedup**, making it highly suitable for real-time robotic control.

---

## Repository Structure

```
.
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── notebooks/
│   └── results_analysis.ipynb   # Jupyter notebook reproducing key results
├── dpfm/                        # Flow Matching implementation
│   ├── config/                  # Hydra configuration files
│   ├── loss/                    # Flow matching loss function
│   ├── sampler/                 # Euler ODE sampler
│   ├── policy/                  # FM policy implementations
│   ├── train.py                 # Training entry point
│   └── eval.py                  # Evaluation entry point
├── diffusion_policy/            # Original Diffusion Policy codebase (baseline)
├── scripts/                     # SLURM job scripts for cluster training
├── results/                     # Evaluation results and metrics
│   ├── eval_baseline/           # DDPM baseline results
│   ├── eval_fm_4step/           # FM 4-step results
│   └── eval_fm_step16/          # FM 16-step results
├── docs/                        # Documentation
│   ├── RESULTS.md               # Detailed experimental results
│   ├── ablation_design.md       # Ablation study design
│   └── final.md                 # Experiment tracking
└── logs/                        # Training logs
```

---

## Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone <repository-url>
cd Diffusion-Policy-Flow-Matching

# Create conda environment
conda create -n DPFM python=3.9 -y
conda activate DPFM

# Install PyTorch (CUDA 11.8)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset

The PushT dataset can be downloaded from the original Diffusion Policy repository:

```bash
# Download PushT dataset (~2GB)
mkdir -p data/training
cd data/training
wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip
unzip pusht.zip
```

### 3. Training

```bash
# Set Python path
export PYTHONPATH="${PWD}:${PWD}/diffusion_policy:$PYTHONPATH"

# Train Flow Matching policy (4 inference steps)
python dpfm/train.py --config-name=train_fm_unet_hybrid_image_workspace \
    policy.num_inference_steps=4 \
    training.seed=42

# Train DDPM baseline
python dpfm/train.py --config-name=train_ddpm_unet_hybrid_pusht \
    training.seed=42
```

### 4. Evaluation

```bash
# Evaluate trained model
python dpfm/eval.py \
    --checkpoint_path <path_to_checkpoint.ckpt> \
    --n_eval_episodes 50 \
    --output_dir results/eval_output
```

### 5. Reproduce Results

Open and run `notebooks/results_analysis.ipynb` to reproduce the key metrics and visualizations.

---

## Method

### Flow Matching vs DDPM

| Aspect | DDPM | Flow Matching |
|--------|------|---------------|
| Forward process | Add Gaussian noise | Linear interpolation |
| Training target | Predict noise ε | Predict velocity v |
| Sampling | 100 DDPM steps | 4-8 Euler ODE steps |
| Latency | ~650 ms | ~25-50 ms |

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
        v = self.model(x, t, obs_encoding)
        x = x + v * dt  # Euler step
    
    return x
```

---

## Ablation Studies

### Effect of Inference Steps

| Steps | Score | Latency | Jerk (smoothness) |
|-------|-------|---------|-------------------|
| 4 | 0.757 | 24 ms | 5576 |
| 8 | **0.779** | 48 ms | 4696 |
| 16 | 0.757 | 151 ms | 4527 |
| 100 (DDPM) | 0.869 | 650 ms | - |

**Insight**: 8 inference steps provides the optimal balance between accuracy and speed.

### Learning Rate Sensitivity

| LR | Best Score | Notes |
|----|------------|-------|
| 1e-4 | 0.779 | Default, stable training |
| 5e-5 | TBD | More stable, slower convergence |
| 1e-3 | TBD | Faster convergence, potentially unstable |

---

## Results Summary

### Completed Experiments (Dec 2025)

| Experiment | Method | Steps | Best Score | Latency (p50) | Epochs |
|------------|--------|-------|------------|---------------|--------|
| ddpm_unet_s42 | DDPM | 100 | 0.869 | 650 ms | ~2050+ (running) |
| fm_unet_s42 | FM | 4 | 0.750 | 24 ms | 1050 |
| fm_steps4 | FM | 4 | 0.757 | 24 ms | 1100 |
| fm_steps8 | FM | 8 | **0.779** | 48 ms | 1300 |
| fm_steps16 | FM | 16 | 0.757 | 151 ms | 1250 |
| fm_trans_s42 | FM Trans | 4 | 0.072 | 21 ms | 1700 |

**Note**: FM Transformer shows poor performance, indicating architecture sensitivity.

---

## Hardware Requirements

- **GPU**: NVIDIA L40S (48GB) or equivalent
- **Memory**: 64-384 GB RAM
- **Storage**: ~400 GB for full dataset and outputs
- **Training Time**: ~6-13 hours per experiment (with early stopping)

---

## WandB Tracking

All experiments are logged to Weights & Biases:

- **Project**: `dpfm_pusht_ablation`
- **Dashboard**: [View Experiments](https://wandb.ai/)

---

## References

1. Chi, C., et al. "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion." RSS 2023. [[Paper](https://arxiv.org/abs/2303.04137)]
2. Lipman, Y., et al. "Flow Matching for Generative Modeling." ICLR 2023. [[Paper](https://arxiv.org/abs/2210.02747)]
3. Liu, X., et al. "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow." ICLR 2023. [[Paper](https://arxiv.org/abs/2209.03003)]

---

## License

This project is for educational purposes (CS 8803). The original Diffusion Policy code is under MIT License.

## Acknowledgments

- [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) authors
- CS 8803 Deep Reinforcement Learning course staff at Georgia Tech
