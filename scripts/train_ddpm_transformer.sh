#!/bin/bash
#SBATCH --job-name=ddpm_trans
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:L40s:1
#SBATCH --time=20:00:00
#SBATCH --partition=gpu-l40s
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --qos=inferno
#SBATCH --output=/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Diffusion-Policy-Flow-Matching/logs/experiments/ddpm_trans_s42_%j.out
#SBATCH --error=/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Diffusion-Policy-Flow-Matching/logs/experiments/ddpm_trans_s42_%j.err

# ============================================================
# Train DDPM Transformer Hybrid on PushT
# ============================================================
# Architecture comparison: DDPM with Transformer backbone
# (complements FM Transformer experiment)

echo "=== DDPM Transformer Training ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Setup
PROJECT_DIR="/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Diffusion-Policy-Flow-Matching"

# Activate environment
source ~/.bashrc
conda activate DPFM

# Environment info
nvidia-smi

# Set paths
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/diffusion_policy:${PYTHONPATH}"
export HYDRA_FULL_ERROR=1

cd ${PROJECT_DIR}/diffusion_policy

echo ""
echo "=========================================="
echo "Training DDPM Transformer (seed=42)"
echo "=========================================="

# Train DDPM with Transformer architecture
python -m dpfm.train \
    --config-name=train_ddpm_transformer_hybrid_pusht \
    exp_name=ddpm_trans_s42 \
    training.seed=42

echo ""
echo "=========================================="
echo "Training Complete"
echo "=========================================="
echo "End time: $(date)"
