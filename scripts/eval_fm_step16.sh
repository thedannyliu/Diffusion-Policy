#!/bin/bash
#SBATCH --job-name=eval_fm16
#SBATCH --output=logs/eval_fm16_%j.out
#SBATCH --error=logs/eval_fm16_%j.err
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --qos=inferno
#SBATCH --exclude=atl1-1-03-004-31-0

# === Evaluate FM Step=16 on PushT ===
# Usage: sbatch scripts/eval_fm_step16.sh

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

source ~/.bashrc
conda activate DPFM

PROJECT_ROOT=/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Diffusion-Policy-Flow-Matching
cd $PROJECT_ROOT
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/diffusion_policy"

# FM Step16 best checkpoint
CHECKPOINT="data/outputs/2025.11.29/00.32.47_train_fm_unet_hybrid_pusht_image/checkpoints/epoch=0650-test_mean_score=0.823.ckpt"

echo "=================================================="
echo "Evaluating FM Step=16"
echo "=================================================="
echo "Checkpoint: $CHECKPOINT"
echo "Start time: $(date)"
echo "=================================================="

nvidia-smi

mkdir -p results/eval_fm_step16

# Run evaluation with 50 test episodes
srun -u python -m dpfm.eval \
    --checkpoint "$CHECKPOINT" \
    --n_test 50 \
    --output_dir results/eval_fm_step16

echo ""
echo "Evaluation completed at: $(date)"
