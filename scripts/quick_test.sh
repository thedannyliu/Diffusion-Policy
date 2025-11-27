#!/bin/bash
#SBATCH --job-name=fm_quick_test
#SBATCH --output=logs/quick_test_%j.out
#SBATCH --error=logs/quick_test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:L40s:1
#SBATCH --time=00:30:00
#SBATCH --partition=gpu-l40s
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --qos=inferno

# Quick test to verify FM Hybrid training works
# Uses debug mode for fast testing

cd /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Diffusion-Policy-Flow-Matching
source ~/.bashrc
conda activate DPFM

export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/diffusion_policy"
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1

echo "=================================================="
echo "Quick Test: FM Hybrid Policy"
echo "=================================================="
echo "Start time: $(date)"
nvidia-smi

# Run with debug mode (only 2 epochs, 3 steps each)
python -m dpfm.train \
    --config-name=train_fm_unet_hybrid_image_workspace \
    training.debug=true \
    exp_name="quick_test" \
    logging.mode=offline

echo ""
echo "Quick test complete!"
echo "End time: $(date)"
