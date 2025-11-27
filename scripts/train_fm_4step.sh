#!/bin/bash
#SBATCH --job-name=dpfm_fm4
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-l40s
#SBATCH --qos=embers
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

# Environment info
nvidia-smi

# Set paths
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/diffusion_policy:${PYTHONPATH}"

# Training
cd ${PROJECT_DIR}/diffusion_policy

python -m dpfm.train \
    --config-name=train_fm_unet_image_workspace \
    policy.num_inference_steps=4 \
    training.seed=42 \
    training.device=cuda:0 \
    exp_name=fm_4step_seed42 \
    logging.mode=online \
    logging.project=dpfm_pusht \
    hydra.run.dir='${PROJECT_DIR}/data/outputs/fm_4step_seed42'

echo "End time: $(date)"
