#!/bin/bash
#SBATCH --job-name=dpfm_fm4
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --partition=gpu-l40s
#SBATCH --output=/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Diffusion-Policy-Flow-Matching/logs/dpfm_fm4_%j.out
#SBATCH --error=/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Diffusion-Policy-Flow-Matching/logs/dpfm_fm4_%j.err

# ============================================================
# Train Flow Matching Policy with 4 Euler steps on Push-T
# ============================================================

echo "=== DPFM Training: Flow Matching 4-step ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Setup
PROJECT_DIR="/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Diffusion-Policy-Flow-Matching"
mkdir -p ${PROJECT_DIR}/logs

# Activate environment
source ~/.bashrc
conda activate DPFM

# Verify packages (don't reinstall if already correct)
python -c "import numpy; assert numpy.__version__.startswith('1.24'), 'numpy version mismatch'" || pip install numpy==1.24.0 --quiet
python -c "import gym" || pip install gym==0.22.0 --quiet

# Environment info
nvidia-smi

# Set paths
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/diffusion_policy:${PYTHONPATH}"
export HYDRA_FULL_ERROR=1

# Training
cd ${PROJECT_DIR}

python -m dpfm.train \
    policy.num_inference_steps=4 \
    training.seed=42 \
    training.device=cuda:0 \
    training.num_epochs=200 \
    exp_name=fm_4step_seed42 \
    logging.mode=offline \
    hydra.run.dir=diffusion_policy/data/outputs/fm_4step_\${now:%Y.%m.%d-%H.%M.%S}

echo "End time: $(date)"
