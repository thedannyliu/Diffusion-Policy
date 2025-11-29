#!/bin/bash
#SBATCH --job-name=ddpm_real_cube
#SBATCH --output=logs/ddpm_real_cube_%j.out
#SBATCH --error=logs/ddpm_real_cube_%j.err
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

# === Optimized DDPM Training for Real Robot (Cube) ===
# Paper-aligned settings for fair comparison with FM
# Usage: sbatch scripts/train_ddpm_real_cube_optimized.sh [seed]

SEED=${1:-42}

echo "=================================================="
echo "DDPM Real Robot Training - Cube"
echo "=================================================="
echo "Seed: $SEED"
echo "Inference Steps: 100 (DDPM)"
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
cd $PROJECT_ROOT/diffusion_policy
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

nvidia-smi

mkdir -p ../logs ../data/outputs/real_robot

srun -u python train.py \
    --config-path=diffusion_policy/config \
    --config-name=train_ddpm_real_robot_workspace \
    task.dataset_path=${PROJECT_ROOT}/data/training/two_cameras_cube \
    task.name=cube \
    training.seed=$SEED \
    training.num_epochs=1000 \
    optimizer.lr=1e-4 \
    logging.project=dpfm_real_robot \
    logging.name="cube_ddpm_seed${SEED}"

echo ""
echo "Training completed at: $(date)"
