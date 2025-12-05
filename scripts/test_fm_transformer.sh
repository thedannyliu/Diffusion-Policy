#!/bin/bash
#SBATCH --job-name=fm_trans_test
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=32GB
#SBATCH --time=00:15:00
#SBATCH --output=logs/fm_trans_test_%j.out
#SBATCH --error=logs/fm_trans_test_%j.err
#SBATCH --partition=gpu-l40s

echo "=========================================="
echo "FM Transformer Quick Test"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "=========================================="

cd /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Diffusion-Policy-Flow-Matching
export PATH="$HOME/miniforge3/bin:$PATH"
source activate DPFM
export PYTHONPATH="${PWD}/diffusion_policy:$PYTHONPATH"

# Quick test: just 10 epochs to see if FM Transformer trains correctly
python dpfm/train.py \
    --config-name=train_fm_transformer_hybrid_image_workspace \
    training.num_epochs=10 \
    training.checkpoint_every=5 \
    training.rollout_every=5 \
    logging.mode=disabled \
    exp_name="fm_trans_test_${SLURM_JOB_ID}"

echo "=========================================="
echo "Test completed!"
echo "=========================================="
