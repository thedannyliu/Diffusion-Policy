#!/bin/bash
#SBATCH --job-name=dpfm_baseline
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-l40s
#SBATCH --qos=embers
#SBATCH --output=/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Diffusion-Policy-Flow-Matching/logs/dpfm_baseline_%j.out
#SBATCH --error=/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Diffusion-Policy-Flow-Matching/logs/dpfm_baseline_%j.err

# ============================================================
# Train DDPM Baseline on Push-T (for fair comparison)
# Uses the original Diffusion Policy implementation
# ============================================================

echo "=== DPFM Training: DDPM Baseline ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Setup
PROJECT_DIR="/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Diffusion-Policy-Flow-Matching"
mkdir -p ${PROJECT_DIR}/logs

# Activate environment
source ~/.bashrc
conda activate DPFM

# Fix package issues  
pip install numpy==1.24.0 --quiet
pip install gym==0.22.0 --quiet

# Environment info
nvidia-smi

# Set paths
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/diffusion_policy:${PYTHONPATH}"
export HYDRA_FULL_ERROR=1

# Training (using original DP)
cd ${PROJECT_DIR}/diffusion_policy

python train.py \
    --config-name=image_pusht_diffusion_policy_cnn \
    training.seed=42 \
    training.device=cuda:0 \
    training.num_epochs=200 \
    logging.mode=offline \
    hydra.run.dir=data/outputs/ddpm_baseline_\${now:%Y.%m.%d-%H.%M.%S}

echo "End time: $(date)"
