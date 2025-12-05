#!/bin/bash
#SBATCH --job-name=test_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:L40s:1
#SBATCH --time=1:00:00
#SBATCH --partition=gpu-l40s
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --qos=inferno
#SBATCH --output=/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Diffusion-Policy-Flow-Matching/logs/test_eval_%j.out
#SBATCH --error=/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Diffusion-Policy-Flow-Matching/logs/test_eval_%j.err

# ============================================================
# Test evaluation with comprehensive metrics
# ============================================================

echo "=== Test Eval ==="
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
python --version
which python

# Set paths
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/diffusion_policy:${PYTHONPATH}"
export HYDRA_FULL_ERROR=1
export MPLBACKEND=Agg  # Use non-interactive backend for matplotlib

cd ${PROJECT_DIR}

# Find latest FM checkpoint
CHECKPOINT=$(find data/outputs -name "latest.ckpt" -path "*train_fm_unet_hybrid_pusht*" 2>/dev/null | sort | tail -1)

if [ -z "$CHECKPOINT" ]; then
    echo "ERROR: No FM checkpoint found!"
    exit 1
fi

echo ""
echo "=========================================="
echo "Testing with checkpoint: $CHECKPOINT"
echo "=========================================="

# Run evaluation with 10 test episodes (quick test)
python -m dpfm.eval \
    --checkpoint "${CHECKPOINT}" \
    --output_dir "results/test_eval_${SLURM_JOB_ID}" \
    --n_test 10 \
    --device cuda:0 \
    --wandb_project "DPFM_eval" \
    --wandb_mode "disabled"

echo ""
echo "=========================================="
echo "Test Complete"
echo "=========================================="
echo "End time: $(date)"
