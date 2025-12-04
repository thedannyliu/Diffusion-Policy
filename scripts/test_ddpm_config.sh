#!/bin/bash
#SBATCH --job-name=test_ddpm
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/test_ddpm_%j.out
#SBATCH --error=logs/test_ddpm_%j.err

echo "=========================================="
echo "Test DDPM Config"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "=========================================="

cd /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Diffusion-Policy-Flow-Matching
export PATH="$HOME/miniforge3/bin:$PATH"
source activate DPFM
export PYTHONPATH="${PWD}/diffusion_policy:$PYTHONPATH"

# Test with just 2 epochs to verify config works
python dpfm/train.py \
    --config-name=train_ddpm_unet_hybrid_pusht \
    exp_name=test_ddpm \
    training.num_epochs=2 \
    training.rollout_every=1 \
    logging.mode=disabled

echo "=========================================="
echo "Test completed: $(date)"
echo "=========================================="
