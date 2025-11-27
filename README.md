# Diffusion Policy with Flow Matching (DPFM)

> **CS 8803 Deep Reinforcement Learning Final Project**: Extending Diffusion Policy with Flow Matching for Faster Inference

## 🎯 Project Overview

This project extends [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) by replacing DDPM with **Flow Matching**, enabling action generation with significantly fewer sampling steps (**4 steps vs 100 steps**) while maintaining comparable task success rates.

### Research Question

> Can Flow Matching training enable Diffusion Policy to achieve comparable success rates with 4 sampling steps, while significantly reducing control latency?

## 📊 Current Results (Push-T Task)

| Method | Steps | Best Score | Latency (p50) | Speedup |
|--------|-------|------------|---------------|---------|
| DDPM (Baseline) | 100 | 0.892 | ~650 ms | 1× |
| FM-DP (step=8) | 8 | 0.845 | ~50 ms | **13×** |
| FM-DP (step=4) | 4 | 0.812 | ~25 ms | **26×** |

> ⚠️ Training still in progress. Results will be updated.

---

## 🏗️ Repository Structure

```
Diffusion-Policy-Flow-Matching/
├── diffusion_policy/          # Original Diffusion Policy codebase (submodule)
│   ├── diffusion_policy/      # Core DP modules
│   ├── data/                  # Datasets (download required)
│   └── train.py               # DDPM training entry point
│
├── dpfm/                      # Our Flow Matching extension
│   ├── config/                # Hydra configs for FM training
│   ├── loss/                  # Flow Matching loss implementation
│   ├── sampler/               # Euler ODE sampler
│   ├── policy/                # FM policy (FlowMatchingUnetHybridImagePolicy)
│   ├── workspace/             # Training workspace
│   └── train.py               # FM training entry point
│
├── scripts/                   # SLURM job scripts (not in git)
├── logs/                      # Training logs (not in git)
├── data/                      # Outputs/checkpoints (not in git)
└── results/                   # Evaluation results (not in git)
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

# Install robomimic (required for hybrid policy)
pip install robomimic==0.2.0 --no-deps
```

### 2. Download Push-T Dataset

```bash
cd diffusion_policy
mkdir -p data && cd data
wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip
unzip pusht.zip && rm pusht.zip
cd ../..
```

---

## 🔬 Training

### Option A: Train DDPM Baseline (Original Diffusion Policy)

```bash
cd diffusion_policy

python train.py \
    --config-path=. \
    --config-name=image_pusht_diffusion_policy_cnn \
    training.seed=42 \
    training.device=cuda:0 \
    logging.mode=online
```

**Key parameters:**
- Config: `diffusion_policy/image_pusht_diffusion_policy_cnn.yaml`
- Epochs: 3050 (paper setting)
- Diffusion steps: 100

### Option B: Train Flow Matching (Our Extension)

```bash
cd /path/to/Diffusion-Policy-Flow-Matching

# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/diffusion_policy"

# Train FM with 4 inference steps
python -m dpfm.train \
    --config-name=train_fm_unet_hybrid_image_workspace \
    policy.num_inference_steps=4 \
    training.seed=42 \
    optimizer.lr=1e-4 \
    logging.mode=online
```

**Key parameters:**
- Config: `dpfm/config/train_fm_unet_hybrid_image_workspace.yaml`
- `policy.num_inference_steps`: Number of Euler steps (2, 4, 8, 16)
- Epochs: 3050 (aligned with DDPM)
- Early stopping: Enabled (patience=10 rollouts)

---

## 📈 Evaluation

### Evaluate a trained checkpoint

```python
import torch
from hydra.utils import instantiate

# Load checkpoint
ckpt = torch.load('path/to/checkpoint.ckpt', map_location='cuda:0')
cfg = ckpt['cfg']

# Instantiate policy
policy = instantiate(cfg.policy)
policy.load_state_dict(ckpt['state_dicts']['ema_model'])
policy.to('cuda:0')
policy.eval()

# Run evaluation using PushTImageRunner
from diffusion_policy.env_runner.pusht_image_runner import PushTImageRunner

runner = PushTImageRunner(
    output_dir='results/',
    n_test=50,
    n_test_vis=4,
    max_steps=300,
    n_obs_steps=cfg.n_obs_steps,
    n_action_steps=cfg.n_action_steps
)

result = runner.run(policy)
print(f"Test mean score: {result['test/mean_score']:.4f}")
```

---

## 🔧 Key Implementation Details

### Flow Matching vs DDPM

| Aspect | DDPM | Flow Matching |
|--------|------|---------------|
| Forward process | Add noise gradually | Linear interpolation |
| Training target | Predict noise ε | Predict velocity v |
| Sampling | 100 DDPM steps | 4 Euler ODE steps |
| Latency | ~650 ms | ~25 ms |

### Flow Matching Loss

```python
# Linear interpolation (Optimal Transport path)
x_t = (1 - t) * noise + t * action  # t ∈ [0, 1]

# Target velocity
v_target = action - noise

# Loss
loss = MSE(v_predicted, v_target)
```

### Euler Sampling

```python
# Start from noise
x = torch.randn_like(action)

# Euler integration with N steps
for i in range(num_steps):
    t = i / num_steps
    v = model(x, t, obs)  # Predict velocity
    x = x + v * (1 / num_steps)  # Euler step

return x  # Final action
```

---

## 📁 Configuration

### FM Config (`dpfm/config/train_fm_unet_hybrid_image_workspace.yaml`)

Key settings aligned with DDPM paper:
```yaml
# Architecture (same as DDPM)
policy:
  down_dims: [512, 1024, 2048]
  crop_shape: [84, 84]
  num_inference_steps: 4  # FM-specific

# Training
training:
  num_epochs: 3050
  lr_warmup_steps: 500
  use_ema: true
  
# Early stopping
  early_stopping: true
  early_stopping_patience: 10  # rollouts
```

---

## 📊 WandB Tracking

All experiments are logged to WandB:
- **Project**: `dpfm_pusht_v2`
- **Metrics**: train_loss, test_mean_score, latency_ms, action_jerk

---

## 🖥️ SLURM (Georgia Tech PACE Phoenix)

Example job script:
```bash
#!/bin/bash
#SBATCH --job-name=fm_train
#SBATCH --partition=gpu-l40s
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --gres=gpu:L40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

conda activate DPFM
python -m dpfm.train --config-name=train_fm_unet_hybrid_image_workspace
```

---

## 📚 References

1. **Diffusion Policy**: Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion", RSS 2023  
   [[Paper](https://arxiv.org/abs/2303.04137)] [[Code](https://github.com/real-stanford/diffusion_policy)]

2. **Flow Matching**: Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023  
   [[Paper](https://arxiv.org/abs/2210.02747)]

3. **Rectified Flow**: Liu et al., "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow", ICLR 2023  
   [[Paper](https://arxiv.org/abs/2209.03003)]

4. **Conditional Flow Matching**: Tong et al., "Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport", 2023  
   [[Paper](https://arxiv.org/abs/2302.00482)]

---

## 📄 License

This project is for educational purposes (CS 8803 Final Project).  
Original Diffusion Policy codebase is under MIT License.

---

## 🙏 Acknowledgments

- [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) authors
- CS 8803 Deep Reinforcement Learning course staff at Georgia Tech
