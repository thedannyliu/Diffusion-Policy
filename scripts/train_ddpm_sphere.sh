#!/bin/bash
#SBATCH --job-name=ddpm_sphere
#SBATCH --output=logs/ddpm_sphere_%j.out
#SBATCH --error=logs/ddpm_sphere_%j.err
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --account=gts-agarg35-ideas_l40s

# Train DDPM baseline on two_cameras_sphere dataset
# Usage: sbatch scripts/train_ddpm_sphere.sh [seed]

export PYTHONUNBUFFERED=1
export WANDB_DIR=/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Diffusion-Policy-Flow-Matching/data/wandb
export WANDB_ENTITY=danny010324
export TQDM_MININTERVAL=5

source ~/.bashrc
conda activate DPFM

PROJECT_ROOT=/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Diffusion-Policy-Flow-Matching
cd $PROJECT_ROOT/diffusion_policy

SEED=${1:-42}

echo "=== Training DDPM Baseline on two_cameras_sphere ==="
echo "Seed: $SEED"
echo "Inference steps: 100 (DDIM)"
echo "Starting at: $(date)"

srun -u python train.py \
    --config-name=train_ddpm_real_robot_workspace \
    task.dataset_path=${PROJECT_ROOT}/data/training/two_cameras_sphere \
    task.name=sphere \
    training.seed=$SEED \
    logging.project=dpfm_real_robot \
    logging.name="sphere_ddpm_seed${SEED}"

echo "Training completed at: $(date)"
