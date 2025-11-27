#!/bin/bash
#SBATCH --job-name=dpfm_eval
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --partition=gpu-l40s
#SBATCH --output=/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Diffusion-Policy-Flow-Matching/logs/eval_%j.out
#SBATCH --error=/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Diffusion-Policy-Flow-Matching/logs/eval_%j.err

# ============================================================
# Evaluate Flow Matching and DDPM policies
# ============================================================

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

# Verify packages
python -c "import numpy; assert numpy.__version__.startswith('1.24'), 'numpy version mismatch'" || pip install numpy==1.24.0 --quiet
python -c "import gym" || pip install gym==0.22.0 --quiet

# Environment info
nvidia-smi

# Set paths
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/diffusion_policy:${PYTHONPATH}"
export HYDRA_FULL_ERROR=1

cd ${PROJECT_DIR}

# Checkpoints
FM_CKPT="diffusion_policy/diffusion_policy/data/outputs/fm_4step_2025.11.26-23.29.05/checkpoints/latest.ckpt"
BASELINE_CKPT="diffusion_policy/data/outputs/ddpm_baseline_2025.11.26-23.31.42/checkpoints/latest.ckpt"

# Evaluate FM 4-step
echo ""
echo "=========================================="
echo "Evaluating Flow Matching (4-step)"
echo "=========================================="
python -m dpfm.eval \
    --checkpoint "${FM_CKPT}" \
    --output_dir "results/eval_fm_4step" \
    --n_test 50 \
    --device cuda:0

# Evaluate Baseline DDPM
echo ""
echo "=========================================="
echo "Evaluating DDPM Baseline (100-step)"
echo "=========================================="
python -m dpfm.eval \
    --checkpoint "${BASELINE_CKPT}" \
    --output_dir "results/eval_baseline" \
    --n_test 50 \
    --device cuda:0

# Summary
echo ""
echo "=========================================="
echo "Evaluation Complete"
echo "=========================================="
echo "FM Results: results/eval_fm_4step/eval_results.json"
echo "Baseline Results: results/eval_baseline/eval_results.json"

echo ""
echo "End time: $(date)"
