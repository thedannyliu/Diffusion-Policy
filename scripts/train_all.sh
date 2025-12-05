#!/bin/bash
# ============================================================================
# Master Training Script for Diffusion Policy Experiments
# ============================================================================
# This script submits all training jobs for:
# 1. DDPM Baselines (UNet, Transformer) - Original Paper Reproduction
# 2. FM Baselines (UNet, Transformer) - Flow Matching with same architecture
# 3. FM Ablations (inference steps, LR/warmup, seeds, data size)
#
# Usage: ./train_all.sh [--dry-run]
#   --dry-run: Print commands without submitting
# ============================================================================

set -e

# Configuration
ACCOUNT="gts-agarg35-ideas_l40s"
PARTITION="gpu-l40s"
TIME="20:00:00"
MEM="384G"
CPUS=8
GPU="L40S:1"

# Project paths
PROJECT_DIR="/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Diffusion-Policy-Flow-Matching"
LOG_DIR="${PROJECT_DIR}/logs"
SCRIPT_DIR="${PROJECT_DIR}/scripts/experiments"

# WandB settings
WANDB_PROJECT="dpfm_pusht_experiments"
WANDB_ENTITY=""  # Set if needed

# Dry run mode
DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE - Commands will be printed but not executed ==="
fi

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$SCRIPT_DIR"

# Track submitted jobs
declare -A JOB_IDS

# ============================================================================
# Helper function to submit a job
# ============================================================================
submit_job() {
    local JOB_NAME="$1"
    local CONFIG="$2"
    local OVERRIDES="$3"
    local SCRIPT_PATH="${SCRIPT_DIR}/${JOB_NAME}.sh"
    
    # Create SLURM script
    cat > "$SCRIPT_PATH" << EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --gres=gpu:${GPU}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_%j.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}_%j.err

echo "=========================================="
echo "Job: ${JOB_NAME}"
echo "SLURM Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURMD_NODENAME"
echo "Start Time: \$(date)"
echo "=========================================="

cd ${PROJECT_DIR}
export PATH="\$HOME/miniforge3/bin:\$PATH"
source activate DPFM
export PYTHONPATH="\${PWD}/diffusion_policy:\$PYTHONPATH"

# Run training
python dpfm/train.py \\
    --config-name=${CONFIG} \\
    ${OVERRIDES}

echo "=========================================="
echo "Job completed: \$(date)"
echo "=========================================="
EOF
    
    chmod +x "$SCRIPT_PATH"
    
    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY RUN] Would submit: $JOB_NAME"
        echo "  Config: $CONFIG"
        echo "  Overrides: $OVERRIDES"
        echo "  Script: $SCRIPT_PATH"
        echo ""
    else
        local JOB_ID=$(sbatch "$SCRIPT_PATH" | awk '{print $4}')
        JOB_IDS["$JOB_NAME"]=$JOB_ID
        echo "Submitted $JOB_NAME (Job ID: $JOB_ID)"
    fi
}

# ============================================================================
# 1. DDPM Baselines (using original diffusion_policy training)
# ============================================================================
echo ""
echo "============================================================================"
echo "1. DDPM BASELINES"
echo "============================================================================"

# DDPM UNet Hybrid (Original Paper Baseline)
submit_job "ddpm_unet_s42" \
    "train_fm_unet_hybrid_image_workspace" \
    "exp_name=ddpm_unet_baseline \
     _target_=diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace.TrainDiffusionUnetHybridWorkspace \
     policy._target_=diffusion_policy.policy.diffusion_unet_hybrid_image_policy.DiffusionUnetHybridImagePolicy \
     policy.noise_scheduler._target_=diffusers.schedulers.scheduling_ddpm.DDPMScheduler \
     policy.noise_scheduler.num_train_timesteps=100 \
     policy.noise_scheduler.beta_schedule=squaredcos_cap_v2 \
     policy.noise_scheduler.clip_sample=true \
     policy.noise_scheduler.prediction_type=epsilon \
     policy.num_inference_steps=100 \
     training.seed=42 \
     logging.project=${WANDB_PROJECT} \
     logging.group=ddpm_baselines \
     logging.tags=[ddpm,unet,baseline,seed42]"

# DDPM Transformer Hybrid
submit_job "ddpm_trans_s42" \
    "train_fm_transformer_hybrid_image_workspace" \
    "exp_name=ddpm_transformer_baseline \
     _target_=diffusion_policy.workspace.train_diffusion_transformer_hybrid_workspace.TrainDiffusionTransformerHybridWorkspace \
     policy._target_=diffusion_policy.policy.diffusion_transformer_hybrid_image_policy.DiffusionTransformerHybridImagePolicy \
     policy.noise_scheduler._target_=diffusers.schedulers.scheduling_ddpm.DDPMScheduler \
     policy.noise_scheduler.num_train_timesteps=100 \
     policy.noise_scheduler.beta_schedule=squaredcos_cap_v2 \
     policy.noise_scheduler.clip_sample=true \
     policy.noise_scheduler.prediction_type=epsilon \
     policy.num_inference_steps=100 \
     training.seed=42 \
     logging.project=${WANDB_PROJECT} \
     logging.group=ddpm_baselines \
     logging.tags=[ddpm,transformer,baseline,seed42]"

# ============================================================================
# 2. FM Baselines
# ============================================================================
echo ""
echo "============================================================================"
echo "2. FM BASELINES"
echo "============================================================================"

# FM UNet Hybrid (4-step Euler)
submit_job "fm_unet_s42" \
    "train_fm_unet_hybrid_image_workspace" \
    "exp_name=fm_unet_baseline \
     policy.num_inference_steps=4 \
     training.seed=42 \
     logging.project=${WANDB_PROJECT} \
     logging.group=fm_baselines \
     logging.tags=[fm,unet,baseline,seed42,steps4]"

# FM Transformer Hybrid (4-step Euler)
submit_job "fm_trans_s42" \
    "train_fm_transformer_hybrid_image_workspace" \
    "exp_name=fm_transformer_baseline \
     policy.num_inference_steps=4 \
     training.seed=42 \
     logging.project=${WANDB_PROJECT} \
     logging.group=fm_baselines \
     logging.tags=[fm,transformer,baseline,seed42,steps4]"

# ============================================================================
# 3. FM Ablation A: Inference Steps
# ============================================================================
echo ""
echo "============================================================================"
echo "3. FM ABLATION A: INFERENCE STEPS"
echo "============================================================================"

for STEPS in 4 8 16; do
    submit_job "fm_unet_steps${STEPS}" \
        "train_fm_unet_hybrid_image_workspace" \
        "exp_name=fm_unet_steps${STEPS} \
         policy.num_inference_steps=${STEPS} \
         training.seed=42 \
         logging.project=${WANDB_PROJECT} \
         logging.group=ablation_inference_steps \
         logging.tags=[fm,unet,ablation,steps${STEPS}]"
done

# ============================================================================
# 4. FM Ablation B: Learning Rate and Warmup
# ============================================================================
echo ""
echo "============================================================================"
echo "4. FM ABLATION B: LR AND WARMUP"
echo "============================================================================"

for LR in "1e-4" "2e-4"; do
    for WARMUP in 500 1000; do
        LR_STR=$(echo $LR | tr '-' 'm')
        submit_job "fm_unet_lr${LR_STR}_w${WARMUP}" \
            "train_fm_unet_hybrid_image_workspace" \
            "exp_name=fm_unet_lr${LR_STR}_w${WARMUP} \
             optimizer.lr=${LR} \
             training.lr_warmup_steps=${WARMUP} \
             policy.num_inference_steps=4 \
             training.seed=42 \
             logging.project=${WANDB_PROJECT} \
             logging.group=ablation_lr_warmup \
             logging.tags=[fm,unet,ablation,lr${LR_STR},warmup${WARMUP}]"
    done
done

# ============================================================================
# 5. FM Ablation C: Seed Sensitivity
# ============================================================================
echo ""
echo "============================================================================"
echo "5. FM ABLATION C: SEED SENSITIVITY"
echo "============================================================================"

for SEED in 42 43 44 45 46; do
    submit_job "fm_unet_seed${SEED}" \
        "train_fm_unet_hybrid_image_workspace" \
        "exp_name=fm_unet_seed${SEED} \
         policy.num_inference_steps=4 \
         training.seed=${SEED} \
         logging.project=${WANDB_PROJECT} \
         logging.group=ablation_seeds \
         logging.tags=[fm,unet,ablation,seed${SEED}]"
done

# ============================================================================
# 6. FM Ablation D: Dataset Size (also run DDPM for comparison)
# ============================================================================
echo ""
echo "============================================================================"
echo "6. FM ABLATION D: DATASET SIZE"
echo "============================================================================"

for DATA_SIZE in 90 60 30; do
    # FM with different data sizes
    submit_job "fm_unet_data${DATA_SIZE}" \
        "train_fm_unet_hybrid_image_workspace" \
        "exp_name=fm_unet_data${DATA_SIZE} \
         task.dataset.max_train_episodes=${DATA_SIZE} \
         policy.num_inference_steps=4 \
         training.seed=42 \
         logging.project=${WANDB_PROJECT} \
         logging.group=ablation_data_size \
         logging.tags=[fm,unet,ablation,data${DATA_SIZE}]"
    
    # DDPM with different data sizes (for comparison)
    submit_job "ddpm_unet_data${DATA_SIZE}" \
        "train_fm_unet_hybrid_image_workspace" \
        "exp_name=ddpm_unet_data${DATA_SIZE} \
         _target_=diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace.TrainDiffusionUnetHybridWorkspace \
         policy._target_=diffusion_policy.policy.diffusion_unet_hybrid_image_policy.DiffusionUnetHybridImagePolicy \
         policy.noise_scheduler._target_=diffusers.schedulers.scheduling_ddpm.DDPMScheduler \
         policy.noise_scheduler.num_train_timesteps=100 \
         policy.noise_scheduler.beta_schedule=squaredcos_cap_v2 \
         policy.noise_scheduler.clip_sample=true \
         policy.noise_scheduler.prediction_type=epsilon \
         policy.num_inference_steps=100 \
         task.dataset.max_train_episodes=${DATA_SIZE} \
         training.seed=42 \
         logging.project=${WANDB_PROJECT} \
         logging.group=ablation_data_size \
         logging.tags=[ddpm,unet,ablation,data${DATA_SIZE}]"
done

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "============================================================================"
echo "SUBMISSION SUMMARY"
echo "============================================================================"

if [[ "$DRY_RUN" == true ]]; then
    echo "DRY RUN - No jobs were submitted"
else
    echo "Submitted jobs:"
    for JOB_NAME in "${!JOB_IDS[@]}"; do
        echo "  $JOB_NAME: ${JOB_IDS[$JOB_NAME]}"
    done
    
    # Save job IDs to file
    JOB_LOG="${LOG_DIR}/submitted_jobs_$(date +%Y%m%d_%H%M%S).txt"
    echo "# Submitted jobs on $(date)" > "$JOB_LOG"
    for JOB_NAME in "${!JOB_IDS[@]}"; do
        echo "$JOB_NAME: ${JOB_IDS[$JOB_NAME]}" >> "$JOB_LOG"
    done
    echo ""
    echo "Job IDs saved to: $JOB_LOG"
fi

echo ""
echo "Total jobs: ${#JOB_IDS[@]}"
echo "============================================================================"
