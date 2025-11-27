#!/bin/bash
#SBATCH --job-name=fm_train
#SBATCH --output=logs/fm_%j.out
#SBATCH --error=logs/fm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:L40s:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-l40s
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --qos=inferno

# === Flow Matching Hybrid Training Script ===
# Aligned with original paper settings (3050 epochs, 84x84 crop, hybrid policy)
# 
# Usage: sbatch scripts/train_fm_hybrid.sh <steps> <seed> <lr> <exp_name>
# Example: sbatch scripts/train_fm_hybrid.sh 4 42 1e-4 step_sweep

# Parse arguments
FM_STEPS=${1:-4}
SEED=${2:-42}
LR=${3:-1e-4}
EXP_NAME=${4:-"fm_hybrid"}

echo "=================================================="
echo "Flow Matching Hybrid Training"
echo "=================================================="
echo "FM Inference Steps: $FM_STEPS"
echo "Seed: $SEED"
echo "Learning Rate: $LR"
echo "Experiment Name: $EXP_NAME"
echo "Start time: $(date)"
echo "=================================================="

# Environment setup
cd /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Diffusion-Policy-Flow-Matching
source ~/.bashrc
conda activate DPFM

# GPU info
nvidia-smi

# Set up paths
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/diffusion_policy"
export WANDB_MODE=online
export HYDRA_FULL_ERROR=1

# Create logs directory
mkdir -p logs

# Run training with clear WandB naming
python -m dpfm.train \
    --config-name=train_fm_unet_hybrid_image_workspace \
    exp_name="${EXP_NAME}" \
    policy.num_inference_steps=${FM_STEPS} \
    training.seed=${SEED} \
    optimizer.lr=${LR} \
    logging.project="dpfm_pusht_v2" \
    logging.group="${EXP_NAME}" \
    logging.name="FM_step${FM_STEPS}_seed${SEED}_lr${LR}"

echo ""
echo "Training complete!"
echo "End time: $(date)"
