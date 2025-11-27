#!/bin/bash
#SBATCH --job-name=ddpm_quick_test
#SBATCH --output=logs/quick_test_ddpm_%j.out
#SBATCH --error=logs/quick_test_ddpm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:L40s:1
#SBATCH --time=00:30:00
#SBATCH --partition=gpu-l40s
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --qos=inferno

# Quick test to verify DDPM baseline training works

cd /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Diffusion-Policy-Flow-Matching
source ~/.bashrc
conda activate DPFM

echo "=================================================="
echo "Quick Test: DDPM Baseline"
echo "=================================================="
echo "Start time: $(date)"
nvidia-smi

# Navigate to diffusion_policy directory
cd diffusion_policy

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1

# Run with debug mode (only 2 epochs)
python train.py \
    --config-path=. \
    --config-name=image_pusht_diffusion_policy_cnn \
    training.debug=true \
    training.seed=42 \
    exp_name="quick_test_ddpm" \
    logging.mode=offline

echo ""
echo "Quick test complete!"
echo "End time: $(date)"
