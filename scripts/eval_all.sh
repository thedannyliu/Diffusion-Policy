#!/bin/bash
#SBATCH --job-name=dpfm_eval_all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:L40s:1
#SBATCH --time=8:00:00
#SBATCH --partition=gpu-l40s
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --qos=inferno
#SBATCH --output=/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Diffusion-Policy-Flow-Matching/logs/eval_all_%j.out
#SBATCH --error=/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Diffusion-Policy-Flow-Matching/logs/eval_all_%j.err

# ============================================================
# Batch Evaluation Script for All Best Checkpoints
# ============================================================
# This script evaluates all best checkpoints from the ablation study
# Results are saved in structured folders under results/

echo "=== DPFM Batch Evaluation ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Setup
PROJECT_DIR="/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Diffusion-Policy-Flow-Matching"
DATA_DIR="${PROJECT_DIR}/data/outputs/2025.12.04"
RESULTS_DIR="${PROJECT_DIR}/results"

mkdir -p ${PROJECT_DIR}/logs
mkdir -p ${RESULTS_DIR}

# Activate environment
source ~/.bashrc
conda activate DPFM

# Environment info
nvidia-smi

# Set paths
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/diffusion_policy:${PYTHONPATH}"
export HYDRA_FULL_ERROR=1

cd ${PROJECT_DIR}/diffusion_policy

# Define all experiments and their best checkpoints
declare -A EXPERIMENTS=(
    ["ddpm_unet_s42"]="${DATA_DIR}/04.06.14_train_ddpm_unet_hybrid_pusht_image/checkpoints/epoch=0450-test_mean_score=0.869.ckpt"
    ["fm_steps16"]="${DATA_DIR}/03.31.47_train_fm_unet_hybrid_pusht_image/checkpoints/epoch=0150-test_mean_score=0.844.ckpt"
    ["fm_lr5e-5"]="${DATA_DIR}/11.23.53_train_fm_unet_hybrid_pusht_image/checkpoints/epoch=0450-test_mean_score=0.820.ckpt"
    ["fm_steps8"]="${DATA_DIR}/03.31.51_train_fm_unet_hybrid_pusht_image/checkpoints/epoch=0150-test_mean_score=0.801.ckpt"
    ["fm_unet_s42"]="${DATA_DIR}/03.31.43_train_fm_unet_hybrid_pusht_image/checkpoints/epoch=0150-test_mean_score=0.777.ckpt"
    ["fm_lr1e-3"]="${DATA_DIR}/17.06.31_train_fm_unet_hybrid_pusht_image/checkpoints/epoch=0600-test_mean_score=0.741.ckpt"
    ["fm_trans_s42"]="${DATA_DIR}/03.31.43_train_fm_transformer_hybrid_pusht_image/checkpoints/epoch=0000-test_mean_score=0.117.ckpt"
)

# Run evaluation for each experiment
for exp_name in "${!EXPERIMENTS[@]}"; do
    checkpoint="${EXPERIMENTS[$exp_name]}"
    output_dir="${RESULTS_DIR}/${exp_name}"
    
    echo ""
    echo "=========================================="
    echo "Evaluating: ${exp_name}"
    echo "Checkpoint: ${checkpoint}"
    echo "Output: ${output_dir}"
    echo "=========================================="
    
    if [ -f "$checkpoint" ]; then
        mkdir -p "${output_dir}"
        python -m dpfm.eval \
            --checkpoint "${checkpoint}" \
            --output_dir "${output_dir}" \
            --n_test 50 \
            --device cuda:0 \
            --wandb_project "DPFM_eval" \
            --wandb_mode "offline"
        
        echo "Completed: ${exp_name}"
    else
        echo "WARNING: Checkpoint not found: ${checkpoint}"
    fi
done

echo ""
echo "=========================================="
echo "All Evaluations Complete"
echo "=========================================="
echo "End time: $(date)"
echo "Results saved in: ${RESULTS_DIR}/"
