#!/bin/bash
#SBATCH --job-name=fm_step16
#SBATCH --output=logs/fm_step16_%j.out
#SBATCH --error=logs/fm_step16_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-l40s
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --qos=inferno
#SBATCH --exclude=atl1-1-03-004-31-0

# === Flow Matching Step=16 Training ===
# Usage: sbatch scripts/train_fm_step16.sh [seed]

SEED=${1:-42}

echo "=================================================="
echo "Flow Matching Step=16 Training"
echo "=================================================="
echo "Seed: $SEED"
echo "Inference Steps: 16"
echo "Start time: $(date)"
echo "=================================================="

cd /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Diffusion-Policy-Flow-Matching
source ~/.bashrc
conda activate DPFM

nvidia-smi

export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/diffusion_policy"
export WANDB_MODE=online
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=0

mkdir -p logs

python -m dpfm.train \
    --config-name=train_fm_unet_hybrid_image_workspace \
    policy.num_inference_steps=16 \
    training.seed=${SEED} \
    optimizer.lr=1e-4 \
    logging.project="dpfm_pusht_v2" \
    logging.name="FM_step16_seed${SEED}"

echo ""
echo "Training complete!"
echo "End time: $(date)"
