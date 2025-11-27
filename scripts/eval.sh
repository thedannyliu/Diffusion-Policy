#!/bin/bash
#SBATCH --job-name=dpfm_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:L40s:1
#SBATCH --time=2:00:00
#SBATCH --partition=gpu-l40s
#SBATCH --account=gts-agarg35
#SBATCH --qos=inferno
#SBATCH --output=/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Diffusion-Policy-Flow-Matching/logs/eval_%j.out
#SBATCH --error=/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Diffusion-Policy-Flow-Matching/logs/eval_%j.err

# ============================================================
# Evaluate Flow Matching and DDPM policies
# ============================================================
# Usage:
#   sbatch scripts/eval.sh                                  # Evaluate both
#   sbatch scripts/eval.sh /path/to/checkpoint.ckpt output_dir  # Evaluate specific checkpoint

CHECKPOINT="${1:-}"
OUTPUT_DIR="${2:-results/eval}"

echo "=== DPFM Evaluation ==="
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
export HYDRA_FULL_ERROR=1

cd ${PROJECT_DIR}/diffusion_policy

# If specific checkpoint provided, evaluate only that
if [ -n "$CHECKPOINT" ]; then
    echo ""
    echo "=========================================="
    echo "Evaluating: $CHECKPOINT"
    echo "=========================================="
    python -m dpfm.eval \
        --checkpoint "${CHECKPOINT}" \
        --output_dir "${OUTPUT_DIR}" \
        --n_test 50 \
        --device cuda:0
else
    # Evaluate all checkpoints in results
    echo "No specific checkpoint provided."
    echo "Please provide checkpoint path: sbatch scripts/eval.sh /path/to/checkpoint.ckpt output_dir"
fi

echo ""
echo "=========================================="
echo "Evaluation Complete"
echo "=========================================="
echo "End time: $(date)"
