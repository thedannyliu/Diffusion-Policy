#!/bin/bash
#SBATCH --job-name=fm_real_sphere
#SBATCH --output=logs/fm_real_sphere_%j.out
#SBATCH --error=logs/fm_real_sphere_%j.err
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --qos=inferno
#SBATCH --exclude=atl1-1-03-004-31-0

# === Optimized FM Training for Real Robot (Sphere) ===
# Paper-aligned settings for fair comparison with DDPM
# Usage: sbatch scripts/train_fm_real_sphere_optimized.sh [seed] [steps]

SEED=${1:-42}
STEPS=${2:-8}

echo "=================================================="
echo "Flow Matching Real Robot Training - Sphere"
echo "=================================================="
echo "Seed: $SEED"
echo "Inference Steps: $STEPS"
echo "Epochs: 1000"
echo "LR: 1e-4"
echo "Start time: $(date)"
echo "=================================================="

export PYTHONUNBUFFERED=1
export WANDB_DIR=/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Diffusion-Policy-Flow-Matching/data/wandb
export WANDB_ENTITY=danny010324
export TQDM_MININTERVAL=5
export HYDRA_FULL_ERROR=1

source ~/.bashrc
conda activate DPFM

PROJECT_ROOT=/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Diffusion-Policy-Flow-Matching
cd $PROJECT_ROOT
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/diffusion_policy"

nvidia-smi

mkdir -p logs data/outputs/real_robot

srun -u python -m dpfm.train \
    --config-name=train_fm_real_robot_fair_workspace \
    task.dataset_path=${PROJECT_ROOT}/data/training/two_cameras_sphere \
    task.name=sphere \
    training.seed=$SEED \
    training.num_epochs=1000 \
    policy.num_inference_steps=$STEPS \
    optimizer.lr=1e-4 \
    exp_name=fm_real_sphere_optimized \
    logging.project=dpfm_real_robot \
    logging.name="sphere_fm_step${STEPS}_seed${SEED}"

echo ""
echo "Training completed at: $(date)"
