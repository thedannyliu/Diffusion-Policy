# Diffusion-Based Visuomotor Policy Learning: Reproduction and Efficiency Analysis

**CS 8803 Deep Reinforcement Learning - Final Project**
**Georgia Institute of Technology - Prof. Animesh Garg**

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![PyTorch 2.0](https://img.shields.io/badge/pytorch-2.0-red.svg)](https://pytorch.org/)

## Abstract

We present a rigorous empirical study of diffusion-based generative modeling for visuomotor policy learning, building upon the Diffusion Policy framework (Chi et al., RSS 2023). This work investigates the computational and performance trade-offs in action-space diffusion models for robotic manipulation through three principal contributions:

**First**, we provide a faithful reproduction of the DDPM-based Diffusion Policy with UNet architecture on the PushT benchmark, establishing properly aligned quantitative metrics (test score, target coverage, success rate) that enable meaningful comparison with published results. Our reproduction achieves 0.816 test score with 0.781 target coverage, validating the baseline implementation.

**Second**, we introduce Flow Matching as an alternative continuous-time generative training objective while maintaining the identical UNet architecture and visual encoders. This architectural consistency enables direct comparison of training objectives, demonstrating that continuous-time optimal transport-based formulations achieve 98% of DDPM task performance while reducing inference latency by 27× (635ms → 24ms) and accelerating training convergence by 3×. The Flow Matching variant serves as a drop-in replacement for the DDPM objective, requiring only modification of the loss function and sampling procedure.

**Third**, we conduct extensive real-world validation using the reproduced DDPM-based policy on a planar non-prehensile manipulation task with a UR10e robotic manipulator. Through systematic ablation studies across observation modalities (monocular vs. stereo), object geometries (cubic vs. spherical), and environmental perturbations, we establish the practical feasibility and limitation boundaries of diffusion-based visuomotor policies in physical systems, achieving 100% success on 32 interior workspace configurations.

### Principal Findings

**Simulation Results (PushT Benchmark):**
- Reproduced DDPM baseline achieves 0.816 test score with 0.781 target coverage, maintaining 100% task success
- Flow Matching attains 98% of DDPM performance (0.798 score, 0.761 coverage) while reducing inference latency by 27× (635ms → 24ms)
- Training convergence accelerated by approximately 3× relative to standard DDPM

**Real-World Deployment (Planar Pushing):**
- Two-camera configuration achieves 100% success rate across 32 evaluation configurations for both rigid and spherical geometries
- Policy exhibits robustness to moderate visual perturbations but degrades under substantial environmental clutter
- Single-camera observations yield significant performance degradation, particularly for rotationally-symmetric objects (50% success for sphere)

---

## Main Results

### Simulation: PushT Benchmark

| Method | Steps | Test Score | Coverage | Latency (p50) | Speedup |
|--------|-------|------------|----------|---------------|---------|
| **DDPM (reproduced)** | 100 | **0.816** | **0.781** | 635.0 ms | 1.0× |
| FM (lr=5e-5) | 4 | 0.798 | 0.761 | 23.1 ms | **27.5×** |
| FM (16 steps) | 16 | 0.794 | 0.758 | 90.4 ms | 7.0× |
| FM (8 steps) | 8 | 0.769 | 0.735 | 45.3 ms | 14.0× |

*All configurations achieve 100% task completion under the evaluation protocol.* Flow Matching with optimized learning rate (5×10⁻⁵) recovers 98% of DDPM performance while providing 27× computational speedup.

### Real-World: Planar Non-Prehensile Manipulation

| Observation Configuration | Object Geometry | Easy Configurations | Hard Configurations | Overall |
|--------------------------|-----------------|--------------------|--------------------|---------|
| Two cameras (stereo) | Cube | 32/32 (100%) | 16/19 (84.2%) | 94.1% |
| Two cameras (stereo) | Sphere | 32/32 (100%) | 11/19 (57.9%) | 84.3% |
| Single camera (monocular) | Cube | 30/32 (93.8%) | — | — |
| Single camera (monocular) | Sphere | 16/32 (50.0%) | — | — |

**Critical Finding**: Stereo observation is necessary for reliable visuomotor control. Policies demonstrate robustness to moderate perceptual noise but exhibit degraded performance under severe environmental clutter.

For detailed ablation results, see [docs/results.md](docs/results.md).

---

## Repository Structure

```
Diffusion-Policy/
├── README.md                     # This document
├── requirements.txt              # Python dependencies
├── notebooks/
│   └── results_analysis.ipynb    # Quantitative analysis and visualization
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
│   ├── ddpm_unet_s42/            # DDPM baseline results
│   ├── fm_steps16/               # FM 16-step results
│   ├── fm_lr5e-5/                # FM with optimized LR
│   └── ...                       # Other experiments
├── logs/                         # Training logs
│   └── experiments/              # SLURM job logs
└── docs/                         # Documentation
    ├── results.md                # Detailed experimental results
    └── ablation_design.md        # Ablation study design
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

## Method

We implement the UNet-based Diffusion Policy architecture and systematically compare two generative modeling objectives for action-space generation. Crucially, both formulations share identical network architectures, visual encoders, and conditioning mechanisms—only the training loss and sampling procedures differ. This design enables direct attribution of performance differences to the choice of generative objective rather than architectural variations.

### Shared Architecture (Both Methods)

Both DDPM and Flow Matching utilize the following components:
- **Vision Encoder**: ResNet18 with spatial softmax pooling (trained end-to-end, no pretraining)
- **State Encoder**: Low-dimensional MLP for proprioceptive observations
- **Temporal Backbone**: 1D Convolutional UNet over action sequences
- **Conditioning**: Feature-wise Linear Modulation (FiLM) to inject observation context
- **Action Representation**: Position control with prediction horizon=16, action horizon=8

### Training Objective 1: DDPM (Baseline Reproduction)

The original Diffusion Policy formulation employs discrete-time denoising:
- Network parameterization: Predict noise $\epsilon_\theta(x_t, t, o)$ at discrete diffusion timesteps
- Training loss: $\mathcal{L}_{\text{DDPM}} = \mathbb{E}_{x_0, \epsilon, t}\|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, t, o)\|^2$
- Inference: Iterative denoising via $x_{t-1} = \alpha(x_t - \gamma\epsilon_\theta(x_t,t,o)) + \mathcal{N}(0,\sigma^2I)$
- Requires 100 denoising steps for convergence at evaluation time

### Training Objective 2: Flow Matching (Our Extension)

We introduce continuous-time flow matching as an alternative, maintaining the same UNet backbone:
- Network parameterization: Predict velocity $v_\theta(x_t, t, o)$ over the action manifold
- Training loss: $\mathcal{L}_{\text{FM}} = \mathbb{E}_{x_1, \epsilon, t}\|v_\theta((1-t)\epsilon + tx_1, t, o) - (x_1 - \epsilon)\|^2$
- Inference: Deterministic ODE integration via $x_{t+\Delta t} = x_t + v_\theta(x_t, t, o)\Delta t$
- Convergence achieved with only 4-16 Euler integration steps

**Key Insight**: Flow Matching serves as a drop-in replacement for DDPM, requiring only modification of the loss function computation during training and replacement of the iterative denoising sampler with an ODE integrator at inference. The UNet architecture, observation encoders, and conditioning mechanisms remain unchanged.

**Comparison:**

| Aspect | DDPM | Flow Matching |
|--------|------|---------------|
| Training target | Predict noise $\epsilon$ | Predict velocity $v$ |
| Sampling | 100 discrete steps | 4-16 continuous steps |
| Inference latency | ~635 ms | ~23-90 ms |
| Training time | ~19 hours | ~6-9 hours |

### Architecture

Both methods use the same **UNet-based** architecture:
- ResNet18 vision encoder (no pretraining)
- Low-dimensional state encoder
- 1D Convolutional UNet for action sequences
- FiLM conditioning for observation integration

**Key implementation details:**
- Position control (not velocity)
- Prediction horizon: 16 steps, Action horizon: 8 steps
- AdamW optimizer with cosine LR schedule
- EMA (exponential moving average) for stable evaluation

---

## Experimental Ablations

We conduct systematic ablation studies to isolate the impact of key design decisions on both computational efficiency and task performance. All experiments are performed under controlled conditions with identical hyperparameters unless explicitly varied.

### 1. ODE Integration Steps (Flow Matching)

| Integration Steps | Test Score | Coverage | Latency (ms) | Relative Speedup |
|------------------|------------|----------|--------------|------------------|
| 4 | 0.798 | 0.761 | 23.1 | **27.5×** |
| 8 | 0.769 | 0.735 | 45.3 | 14.0× |
| 16 | 0.794 | 0.758 | 90.4 | 7.0× |
| 100 (DDPM) | 0.816 | 0.781 | 635.0 | 1.0× |

**Analysis**: Flow Matching exhibits relative insensitivity to the number of Euler integration steps across the 4-16 range, with performance variance <4%. This suggests that the learned velocity field provides sufficient smoothness for accurate trajectory integration with minimal discretization. The optimal operating point (4 steps) achieves 98% of DDPM performance while enabling real-time control at 40+ Hz.

### 2. Learning Rate Hyperparameter

| Learning Rate | Test Score | Coverage | Training Stability |
|--------------|------------|----------|-------------------|
| 1×10⁻⁴ (default) | 0.667 | 0.637 | Baseline |
| **5×10⁻⁵** | **0.798** | **0.761** | Enhanced convergence |
| 1×10⁻³ | 0.666 | 0.637 | Optimization instability |

**Analysis**: Flow Matching demonstrates marked sensitivity to learning rate selection, with a 20% performance improvement under conservative optimization (5×10⁻⁵). This contrasts with DDPM's relative robustness to learning rate variation, suggesting that continuous-time formulations require more careful tuning of the optimization landscape. Aggressive learning rates (1×10⁻³) induce training instability without performance gains.

### 3. Network Architecture

| Backbone | Training Objective | Test Score | Convergence |
|----------|-------------------|------------|-------------|
| UNet | DDPM | 0.816 | Stable |
| UNet | Flow Matching | 0.798 | Stable |
| Transformer | Flow Matching | 0.114 | Failed |

**Analysis**: The convolutional UNet architecture with Feature-wise Linear Modulation (FiLM) conditioning demonstrates consistent efficacy across both DDPM and Flow Matching objectives. In stark contrast, Transformer-based temporal models fail catastrophically under Flow Matching training (0.114 test score ≈ random policy). This architectural dependency suggests that local temporal inductive biases inherent to convolutions may be crucial for learning smooth velocity fields in action space.

### 4. Observation Modality (Real-World)

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

### Principal Findings

This empirical investigation establishes several key insights regarding diffusion-based visuomotor policy learning:

**Reproducibility and Metric Alignment**: We successfully reproduced the DDPM-based Diffusion Policy with UNet architecture on the PushT benchmark, achieving a test score of 0.816 with target coverage of 0.781 and 100% task completion rate. This reproduction, aligned with the original evaluation protocol, provides a validated baseline implementation that serves as the foundation for both our Flow Matching extension and real-world deployment.

**Computational Efficiency via Flow Matching**: Building upon our reproduced DDPM baseline, we introduce continuous-time Flow Matching as an alternative training objective using the identical UNet architecture. This architectural consistency enables direct comparison, demonstrating that optimal transport-based formulations achieve substantial computational gains (27× inference speedup, 3× training acceleration) while maintaining 98% of DDPM task performance in simulation. The Flow Matching extension requires only loss function modification, serving as a drop-in replacement that suggests the discrete-time diffusion framework may be over-parameterized for action-space generation.

**Observation Modality Requirements**: Real-world deployment reveals that stereo visual observation is not merely beneficial but necessary for reliable visuomotor control, particularly for objects with rotational symmetries. Monocular configurations exhibit catastrophic performance degradation (50% success for spherical objects), highlighting the critical role of depth perception in contact-rich manipulation.

**Geometric Task Complexity**: Cubic object manipulation exhibits higher failure rates (84% vs. 100% success in boundary configurations) compared to spherical objects, attributable to the requirement for precise angular alignment during slot insertion. This observation suggests that diffusion policies may benefit from explicit geometric reasoning mechanisms.

**Hyperparameter Sensitivity**: Flow Matching exhibits marked sensitivity to learning rate selection, with conservative values (5×10⁻⁵) yielding 20% performance improvement over standard DDPM settings (1×10⁻⁴). This suggests that the continuous-time formulation requires more careful optimization tuning than its discrete-time counterpart.

**Architectural Constraints**: The convolutional UNet backbone proves effective for both DDPM and Flow Matching objectives, while Transformer-based architectures fail to converge (0.114 test score). This architectural dependency warrants further investigation into inductive biases suitable for action-space generation.

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
