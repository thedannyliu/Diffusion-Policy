# Diffusion Policy: Reproduction and Real-World Validation

**CS 8803 Deep Reinforcement Learning - Final Project**
**Georgia Institute of Technology - Prof. Animesh Garg**

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![PyTorch 2.0](https://img.shields.io/badge/pytorch-2.0-red.svg)](https://pytorch.org/)

## Overview

This project presents a comprehensive empirical study of **Diffusion Policy** (Chi et al., RSS 2023) for visuomotor policy learning in robotic manipulation. We provide a faithful reproduction of the original method, validate it extensively in real-world scenarios, and introduce an efficient Flow Matching variant as an optional improvement.

### Three Principal Contributions

**1. Baseline Reproduction & Validation**
We faithfully reproduce the DDPM-based Diffusion Policy with UNet architecture on the PushT simulation benchmark, establishing properly aligned quantitative metrics (test score, target coverage, success rate). Our reproduction achieves **0.816 test score** with **0.781 target coverage**, validating the baseline implementation and providing a solid foundation for further work.

**2. Real-World Deployment & Ablation Studies**
We conduct extensive real-world validation using a **UR10e robotic manipulator** on planar non-prehensile manipulation tasks. Through systematic ablation studies across observation modalities (monocular vs. stereo), object geometries (cube vs. sphere), and environmental perturbations, we establish practical feasibility boundaries of diffusion-based visuomotor policies. The two-camera stereo configuration achieves **100% success** across 32 interior workspace configurations for both object types.

**3. Flow Matching Extension (Optional Efficiency Improvement)**
As an additional contribution, we introduce Flow Matching as an alternative continuous-time training objective while maintaining identical network architecture. This variant achieves **98% of baseline performance** while reducing inference latency by **27×** (635ms → 24ms) and accelerating training by **3×**, serving as a drop-in replacement requiring only loss function and sampling procedure modifications.

### Key Findings

**Real-World Robot Deployment:**
- Two-camera stereo configuration achieves **100% success** across 32 interior workspace positions for both cube and sphere objects
- **Stereo vision is essential**: Single-camera setup degrades to 50% success for spheres and 93.8% for cubes
- Policy demonstrates robustness to moderate visual perturbations but degrades under heavy environmental clutter
- Cube manipulation requires more precise angular alignment than sphere (84.2% vs 57.9% success on boundary configurations)

**Simulation Reproduction (PushT Benchmark):**
- Successfully reproduced DDPM Diffusion Policy: **0.816 test score**, **0.781 target coverage**, 100% task success
- Validated metrics align with original published results, confirming faithful implementation

**Flow Matching Efficiency Gains (Optional):**
- Achieves 98% of DDPM performance (0.798 score, 0.761 coverage) with only 4 ODE integration steps
- **27× faster inference** (635ms → 24ms) enabling real-time control at 40+ Hz
- **3× faster training** convergence compared to standard DDPM

---

## Main Results

### Real-World: Planar Non-Prehensile Manipulation

| Observation Configuration | Object Geometry | Easy Spots (32) | Hard Spots (18) | Overall |
|--------------------------|-----------------|-----------------|-----------------|---------|
| **Two cameras (stereo)** | **Cube** | **32/32 (100%)** | 16/19 (84.2%) | 94.1% |
| **Two cameras (stereo)** | **Sphere** | **32/32 (100%)** | 11/19 (57.9%) | 84.3% |
| Single camera (monocular) | Cube | 30/32 (93.8%) | — | — |
| Single camera (monocular) | Sphere | 16/32 (50.0%) | — | — |

**Key Insight**: Stereo observation is essential for reliable visuomotor control in contact-rich manipulation. Single-camera configurations suffer catastrophic performance degradation for rotationally-symmetric objects (sphere: 50% vs 100% success).

**Hardware**: UR10e robot with Robotiq 85 gripper, dual Intel RealSense D435 cameras, custom-designed task board with 3D-printed fixtures, SpaceMouse teleoperation for 100 demonstrations per task.

### Simulation: PushT Benchmark Reproduction

**DDPM Baseline (Reproduced):**
- Test Score: **0.816** | Coverage: **0.781** | Success: **100%**
- Inference: 635ms per step (100 denoising steps)
- Training: ~19 hours on A40 GPU

**Flow Matching Variant (Optional Efficiency Improvement):**

| Configuration | Steps | Test Score | Coverage | Latency (p50) | Speedup |
|--------------|-------|------------|----------|---------------|---------|
| **FM (lr=5e-5)** | **4** | **0.798** | **0.761** | **23.1 ms** | **27.5×** |
| FM (16 steps) | 16 | 0.794 | 0.758 | 90.4 ms | 7.0× |
| FM (8 steps) | 8 | 0.769 | 0.735 | 45.3 ms | 14.0× |
| FM (default lr) | 4 | 0.667 | 0.637 | 23.1 ms | 27.5× |

*Flow Matching with optimized learning rate (5×10⁻⁵) achieves 98% of DDPM performance while enabling real-time control at 40+ Hz.*

For detailed ablation results, see [docs/results.md](docs/results.md).

---

## Repository Structure

```
Diffusion-Policy/
├── README.md                     # This document
├── report.pdf                    # Full technical report
├── requirements.txt              # Python dependencies
│
├── diffusion_policy/             # Core implementation (DDPM baseline)
│   ├── policy/                   # Policy architectures (UNet, Transformer)
│   ├── model/                    # Vision encoders, diffusion models
│   ├── dataset/                  # Data loading and preprocessing
│   └── ...                       # Training, evaluation utilities
│
├── dpfm/                         # Extension: Flow Matching variant
│   ├── config/                   # Training configurations
│   │   ├── train_ddpm_unet_hybrid_pusht.yaml      # DDPM baseline config
│   │   └── train_fm_unet_hybrid_image_workspace.yaml  # Flow Matching config
│   ├── loss/                     # Flow matching loss (CFM)
│   ├── sampler/                  # Euler ODE sampler
│   ├── policy/                   # FM policy wrapper
│   ├── train.py                  # Training script
│   └── eval.py                   # Evaluation script
│
├── data/                         # Datasets (download separately)
│   ├── training/
│   │   ├── pusht/                # PushT simulation dataset
│   │   ├── two_cameras_cube/     # Real robot cube demos
│   │   └── two_cameras_sphere/   # Real robot sphere demos
│   └── evaluations/              # Evaluation rollout data
│
├── notebooks/
│   └── results_analysis.ipynb    # Result visualization and analysis
│
├── scripts/                      # Job submission scripts
├── results/                      # Evaluation outputs
└── docs/                         # Additional documentation
```

---

## Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/thedannyliu/Diffusion-Policy.git
cd Diffusion-Policy

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

#### Real Robot Data & Plotting

For training the diffusion policy from real collected demonstrations we used the following:

```bash
Download data.zip (~2.5GB) from Google Drive
# https://drive.google.com/drive/folders/1fy_uOMjU0OhQmbbqVIfamUVsaG92qr6L?usp=sharing

In this folder you cand find our training data, visualizations and record of the evaluations/abllations conducted. 

# Extract to repo root
unzip data.zip

# This creates:
# - data/training/        (two_cameras_cube, two_cameras_sphere)
# - data/evaluations/     (9 evaluation runs)
```

**Optional**: Download pre-generated `Trajectory_Plots/` folder from the drive.

See **[data/README.md](data/README.md)** for detailed setup instructions and plotting script usage.

### 3. Training

```bash
# Set Python path
export PYTHONPATH="${PWD}:${PWD}/diffusion_policy:$PYTHONPATH"
cd diffusion_policy

# Train DDPM baseline (original Diffusion Policy, 100 inference steps)
python dpfm/train.py --config-name=train_ddpm_unet_hybrid_pusht \
    training.seed=42

# OPTIONAL: Train Flow Matching variant for faster inference
# Flow Matching with optimized learning rate (recommended)
python dpfm/train.py --config-name=train_fm_unet_hybrid_image_workspace \
    policy.num_inference_steps=4 \
    optimizer.lr=5e-5 \
    training.seed=42

# Flow Matching with default settings
python dpfm/train.py --config-name=train_fm_unet_hybrid_image_workspace \
    policy.num_inference_steps=4 \
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

## Method

We implement and reproduce the **Diffusion Policy** architecture (Chi et al., RSS 2023) for visuomotor control, deploying it on both simulation and real-world robotic manipulation tasks. Additionally, we introduce an optional **Flow Matching** variant that achieves comparable performance with significantly reduced computational cost.

### Core Architecture: Diffusion Policy (DDPM)

The baseline method uses a UNet-based conditional diffusion model for action-space generation:

**Architecture Components:**
- **Vision Encoder**: ResNet18 with spatial softmax pooling (trained end-to-end, no pretraining)
- **State Encoder**: Low-dimensional MLP for proprioceptive observations (robot joint states)
- **Temporal Backbone**: 1D Convolutional UNet operating over action sequences
- **Conditioning**: Feature-wise Linear Modulation (FiLM) to inject observation context into denoising
- **Action Representation**: Position control with prediction horizon=16, action horizon=8

**Training Objective (DDPM):**
- Network learns to predict noise $\epsilon_\theta(x_t, t, o)$ at discrete diffusion timesteps
- Training loss: $\mathcal{L}_{\text{DDPM}} = \mathbb{E}_{x_0, \epsilon, t}\|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, t, o)\|^2$
- Inference: Iterative denoising via $x_{t-1} = \alpha(x_t - \gamma\epsilon_\theta(x_t,t,o)) + \mathcal{N}(0,\sigma^2I)$
- Requires 100 denoising steps for high-quality action generation

**Implementation Details:**
- Position control (not velocity) for stability
- AdamW optimizer with cosine learning rate schedule
- Exponential Moving Average (EMA) for stable policy evaluation
- Training time: ~12-19 hours on A40/L40S GPU

### Optional Extension: Flow Matching

As an efficiency improvement, we introduce continuous-time flow matching while maintaining the **identical UNet architecture**:

**Modified Training Objective:**
- Network predicts velocity field $v_\theta(x_t, t, o)$ over action manifold instead of noise
- Training loss: $\mathcal{L}_{\text{FM}} = \mathbb{E}_{x_1, \epsilon, t}\|v_\theta((1-t)\epsilon + tx_1, t, o) - (x_1 - \epsilon)\|^2$
- Inference: Deterministic ODE integration via $x_{t+\Delta t} = x_t + v_\theta(x_t, t, o)\Delta t$
- Only 4-16 Euler integration steps needed (vs 100 for DDPM)

**Key Insight**: Flow Matching is a drop-in replacement requiring only loss function and sampling procedure changes—the UNet architecture, visual encoders, and conditioning mechanisms remain identical to DDPM.

**Performance Comparison:**

| Aspect | DDPM (Baseline) | Flow Matching (Ours) |
|--------|-----------------|----------------------|
| Training target | Predict noise $\epsilon$ | Predict velocity $v$ |
| Inference steps | 100 denoising steps | 4-16 ODE steps |
| Inference latency | ~635 ms/step | ~23-90 ms/step |
| Training time | ~19 hours | ~6-9 hours |
| PushT test score | 0.816 | 0.798 (98% of baseline) |

---

## Experimental Ablations

We conduct systematic ablation studies to evaluate both the reproduced DDPM baseline and the Flow Matching extension. All experiments use controlled conditions with identical hyperparameters unless explicitly varied.

### Flow Matching Ablations (Simulation Only)

The following ablations study the Flow Matching variant in PushT simulation. Real-world experiments use the DDPM baseline.

#### 1. ODE Integration Steps

| Integration Steps | Test Score | Coverage | Latency (ms) | Relative Speedup |
|------------------|------------|----------|--------------|------------------|
| 4 | 0.798 | 0.761 | 23.1 | **27.5×** |
| 8 | 0.769 | 0.735 | 45.3 | 14.0× |
| 16 | 0.794 | 0.758 | 90.4 | 7.0× |
| 100 (DDPM) | 0.816 | 0.781 | 635.0 | 1.0× |

**Analysis**: Flow Matching exhibits relative insensitivity to the number of Euler integration steps across the 4-16 range, with performance variance <4%. This suggests that the learned velocity field provides sufficient smoothness for accurate trajectory integration with minimal discretization. The optimal operating point (4 steps) achieves 98% of DDPM performance while enabling real-time control at 40+ Hz.

#### 2. Learning Rate Sensitivity

| Learning Rate | Test Score | Coverage | Training Stability |
|--------------|------------|----------|-------------------|
| 1×10⁻⁴ (default) | 0.667 | 0.637 | Baseline |
| **5×10⁻⁵** | **0.798** | **0.761** | Enhanced convergence |
| 1×10⁻³ | 0.666 | 0.637 | Optimization instability |

**Analysis**: Flow Matching demonstrates marked sensitivity to learning rate selection, with a 20% performance improvement under conservative optimization (5×10⁻⁵). This contrasts with DDPM's relative robustness to learning rate variation, suggesting that continuous-time formulations require more careful tuning of the optimization landscape. Aggressive learning rates (1×10⁻³) induce training instability without performance gains.

#### 3. Architecture Comparison (UNet vs Transformer)

| Backbone | Training Objective | Test Score | Convergence |
|----------|-------------------|------------|-------------|
| UNet | DDPM | 0.816 | Stable |
| UNet | Flow Matching | 0.798 | Stable |
| Transformer | Flow Matching | 0.114 | Failed |

**Analysis**: The convolutional UNet architecture with Feature-wise Linear Modulation (FiLM) conditioning demonstrates consistent efficacy across both DDPM and Flow Matching objectives. In stark contrast, Transformer-based temporal models fail catastrophically under Flow Matching training (0.114 test score ≈ random policy). This architectural dependency suggests that local temporal inductive biases inherent to convolutions may be crucial for learning smooth velocity fields in action space.

### Real-World Ablations (DDPM Baseline)

The following ablations evaluate the reproduced DDPM Diffusion Policy on real robot hardware (UR10e).

#### 4. Observation Modality: Monocular vs Stereo

| Visual Configuration | Cube Success | Sphere Success | Geometric Robustness |
|---------------------|--------------|----------------|---------------------|
| Stereo (two cameras) | 100% | 100% | Complete |
| Monocular (single camera) | 93.8% | 50.0% | Degraded |

**Analysis**: Stereo visual observation proves essential for reliable manipulation, particularly for rotationally-symmetric objects. The 50% performance degradation for spherical objects under monocular observation indicates that depth perception is critical for accurate contact positioning in non-prehensile manipulation. The asymmetry between cube (93.8%) and sphere (50%) performance under monocular observation suggests that geometric features with directional cues provide partial compensation for missing depth information.

---

## Data Format

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

## Real-World Experimental Setup

To validate the practical applicability of diffusion-based visuomotor policies beyond simulation, we design and execute a systematic real-world evaluation on a planar non-prehensile manipulation task. Our experimental protocol emphasizes reproducibility and controlled variation of environmental factors.

### Hardware Configuration

**Robotic Platform**: Universal Robots UR10e collaborative manipulator equipped with Robotiq 85 adaptive parallel-jaw gripper. The UR10e provides 6-DOF workspace coverage with ±0.1mm repeatability, enabling precise positioning for contact-based manipulation.

**Perception System**: Dual Intel RealSense D435 RGB-D cameras positioned at complementary viewpoints (frontal and lateral) to provide stereo visual observation. RGB channels exclusively utilized (depth discarded) to match simulation-trained policy input modality.

**Teleoperation Interface**: 3Dconnexion SpaceMouse for kinesthetic demonstration collection, providing intuitive 6-DOF Cartesian control with vertical axis fixed to maintain planar constraint.

**Custom Fixtures**: Task board, camera mounts, and slot inserts designed in Onshape and fabricated via FDM 3D printing, ensuring geometric consistency across data collection and evaluation episodes.

### Task Specification

**Objective**: Planar non-prehensile pushing to transport objects from arbitrary workspace configurations into designated goal slots via contact manipulation.

**Object Geometries**:
- Cubic: 30mm × 30mm base, 40mm height (introduces angular alignment requirements)
- Spherical: 30mm diameter, 40mm height (rotationally symmetric, minimal alignment constraints)

**Goal Specification**: 40mm × 40mm internal slot dimensions, providing 10mm tolerance envelope around object base to admit slight positioning errors while maintaining non-trivial insertion requirements.

### Demonstration Collection Protocol

**Dataset Scale**: 100 kinesthetic demonstrations per object geometry (200 total trajectories)

**Spatial Coverage**: 50-position grid spanning workspace (5×10 lattice), partitioned into:
- 32 "easy" configurations (interior region with direct paths to goal)
- 18 "hard" configurations (boundary positions requiring longer, obstacle-aware trajectories)

**Rotational Augmentation**: For cubic objects, two collection cycles with distinct orientation distributions:
1. First cycle: Fixed canonical orientation
2. Second cycle: Uniformly random in-plane rotations

This augmentation strategy encourages learning rotation-invariant pushing strategies rather than memorizing orientation-specific contact points.

### Evaluation Protocol

**Success Criterion**: Binary classification based on physical insertion into goal slot (ground truth determined by mechanical contact, independent of visual observation)

**Episode Constraints**:
- 30-second temporal budget per trial
- Standardized reset procedure to eliminate configuration drift
- Robot initialization to fixed home pose in base frame coordinates

**Metric Computation**: Success rate aggregated across spatial configurations, stratified by difficulty (easy vs. hard) and object geometry to isolate task complexity factors.

## IoU using Segmentation
It's recommended to create a separate conda environment to evaluate IoU.
```
conda create -n seg python==3.10.0
conda activate seg
cd segment-anything-annotator
pip install -r requirements.txt
```

Then, run the following
```
python segment-anything-annotator/diffusion/calculate_iou.py
```

**Note**: IoU found to be unreliable metric for small objects due to camera pose and segmentation sensitivity. Physical success (slot insertion) used as primary metric.

---

## Conclusions

### Summary of Contributions

This work presents a comprehensive empirical study of Diffusion Policy through three main contributions:

**1. Faithful Baseline Reproduction**

We successfully reproduced the DDPM-based Diffusion Policy with UNet architecture on the PushT benchmark, achieving **test score 0.816** with **target coverage 0.781** and 100% success rate. This reproduction, aligned with the original evaluation protocol and metrics, provides a validated baseline that served as the foundation for real-world deployment and comparative analysis.

**2. Extensive Real-World Validation**

We conducted systematic real-world evaluation on a UR10e robotic manipulator performing planar non-prehensile manipulation with custom-designed hardware. Key findings:

- **Stereo vision is essential**: Two-camera configuration achieves 100% success on 32 interior workspace positions for both cube and sphere objects
- **Monocular observation catastrophically degrades performance**: Single-camera setup drops to 50% success for spheres (vs 100% for stereo)
- **Robustness to moderate perturbations**: Policy handles gripper appearance changes and limited clutter but degrades under heavy environmental disturbances
- **Geometric complexity matters**: Cubes require more precise angular alignment than spheres (84% vs 58% success on boundary configurations)

**3. Flow Matching Efficiency Improvement (Optional)**

As an additional contribution, we introduce continuous-time Flow Matching as a drop-in replacement for DDPM, using the identical UNet architecture but replacing only the loss function and sampling procedure:

- Achieves **98% of baseline performance** (0.798 vs 0.816 test score)
- **27× faster inference** (24ms vs 635ms) enabling real-time control at 40+ Hz
- **3× faster training** convergence
- Demonstrates that discrete-time diffusion may be over-parameterized for action-space generation

**Other Key Findings:**

- **Architecture matters**: UNet succeeds for both DDPM and Flow Matching; Transformer fails catastrophically with Flow Matching (0.114 test score)
- **Hyperparameter sensitivity**: Flow Matching requires careful learning rate tuning (5×10⁻⁵ optimal vs 1×10⁻⁴ default)

### Limitations and Future Research Directions

**Statistical Rigor**: The majority of our experiments utilize single random seeds due to computational constraints. Multi-seed evaluation across diverse initialization conditions would strengthen the statistical validity of our comparative claims.

**Real-World Flow Matching Deployment**: Our real-world validation utilizes the reproduced DDPM baseline policy, while the Flow Matching variant remains evaluated exclusively in PushT simulation. Given that Flow Matching maintains the identical UNet architecture and requires only training objective modification, real robot deployment would provide valuable evidence regarding whether the 27× inference speedup translates to improved closed-loop control performance in physical systems with sensor latencies and actuation delays.

**Architectural Exploration**: The failure of Transformer backbones under Flow Matching training suggests that attention-based architectures may require specialized normalization schemes or hybrid designs combining convolutional inductive biases with long-range dependencies.

**Domain Complexity**: Our evaluation focuses on planar non-prehensile manipulation—a constrained domain with reduced state dimensionality. Extension to contact-rich grasping, multi-step task composition, or deformable object manipulation would assess the scalability of diffusion-based policy learning to higher-complexity manipulation problems.

**Theoretical Understanding**: The empirical success of Flow Matching over DDPM raises theoretical questions regarding the role of discrete vs. continuous-time formulations in learning action distributions. Formal analysis of approximation error, sample complexity, and convergence guarantees would provide principled guidance for objective selection.

---

## References

1. Chi, C., et al. "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion." RSS 2023. [[Paper](https://arxiv.org/abs/2303.04137)] [[Code](https://github.com/real-stanford/diffusion_policy)]

2. Lipman, Y., et al. "Flow Matching for Generative Modeling." ICLR 2023. [[Paper](https://arxiv.org/abs/2210.02747)]

3. Liu, X., et al. "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow." ICLR 2023. [[Paper](https://arxiv.org/abs/2209.03003)]

---

## License

This project is for educational purposes (CS 8803 Deep Reinforcement Learning, Georgia Tech). The original Diffusion Policy code is under MIT License.

## Acknowledgments

We gratefully acknowledge the following contributions to this work:

- **Cheng Chi, Siyuan Feng, Yilun Du, Zhenjia Xu, Eric Cousineau, Benjamin Burchfiel, and Shuran Song** for developing the original Diffusion Policy framework and releasing their codebase, which served as the foundation for our reproduction and extension.

- **Professor Animesh Garg** and the CS 8803 Deep Reinforcement Learning teaching staff at Georgia Institute of Technology for guidance on experimental design and rigorous empirical evaluation.

- **Georgia Tech Partnership for an Advanced Computing Environment (PACE)** for providing computational resources essential for large-scale hyperparameter sweeps and multi-seed training runs.

- **Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le** for their foundational work on Flow Matching for Generative Modeling (ICLR 2023), which inspired our continuous-time alternative formulation.
