#!/bin/bash
#SBATCH --job-name=test_fm_load
#SBATCH --output=logs/test_fm_load_%j.out
#SBATCH --error=logs/test_fm_load_%j.err
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --qos=inferno
#SBATCH --exclude=atl1-1-03-004-31-0

# === Test FM Checkpoint Loading ===
# This script tests that FM checkpoints can be loaded correctly
# and that the policy interface works as expected.

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

source ~/.bashrc
conda activate DPFM

PROJECT_ROOT=/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Diffusion-Policy-Flow-Matching
cd $PROJECT_ROOT
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/diffusion_policy"

echo "=================================================="
echo "Testing FM Checkpoint Loading"
echo "=================================================="
echo "Start time: $(date)"
echo "=================================================="

nvidia-smi

# Test loading FM checkpoints
python scripts/test_fm_checkpoint_loading.py

echo ""
echo "Test completed at: $(date)"
