#!/bin/bash
#SBATCH --job-name=ddpm_train
#SBATCH --output=logs/ddpm_%j.out
#SBATCH --error=logs/ddpm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:L40s:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-l40s
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --qos=inferno

# === DDPM Baseline Hybrid Training Script ===
# Uses original paper config: diffusion_policy/image_pusht_diffusion_policy_cnn.yaml
# This trains the DiffusionUnetHybridImagePolicy for fair comparison

# Parse arguments
SEED=${1:-42}

echo "=================================================="
echo "DDPM Baseline Hybrid Training"
echo "=================================================="
echo "Seed: $SEED"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=================================================="

# Environment setup
cd /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Diffusion-Policy-Flow-Matching
source ~/.bashrc
conda activate DPFM

# GPU info
nvidia-smi

# Navigate to diffusion_policy directory
cd diffusion_policy

# Set up paths
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export WANDB_MODE=online
export HYDRA_FULL_ERROR=1

# Run training using paper config
python train.py \
    --config-path=. \
    --config-name=image_pusht_diffusion_policy_cnn \
    training.seed=${SEED} \
    exp_name="ddpm_baseline_s${SEED}" \
    logging.project="dpfm_pusht_v2" \
    logging.group="baseline" \
    logging.name="DDPM_seed${SEED}" \
    logging.mode="online"

echo ""
echo "Baseline training complete!"
echo "End time: $(date)"
