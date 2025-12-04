#!/bin/bash
# ============================================================================
# Comprehensive Training Script for DDPM and FM Experiments
# ============================================================================
# Submits training jobs for:
# - DDPM baselines (UNet, Transformer)
# - FM baselines (UNet, Transformer)
# - FM ablations (inference steps, LR/warmup, seeds, data efficiency)
#
# Usage: bash scripts/submit_experiments.sh [--dry-run]
# ============================================================================

set -e

PROJECT_DIR="/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Diffusion-Policy-Flow-Matching"
cd "$PROJECT_DIR"

# SLURM settings
ACCOUNT="gts-agarg35-ideas_l40s"
PARTITION="gpu-l40s"
TIME="20:00:00"
MEM="384G"
CPUS=8
GPU="L40S:1"

LOG_DIR="${PROJECT_DIR}/logs/experiments"
mkdir -p "$LOG_DIR"

DRY_RUN=false
[[ "$1" == "--dry-run" ]] && DRY_RUN=true && echo "=== DRY RUN MODE ==="

# Job ID tracking file
JOB_LOG="${LOG_DIR}/jobs_$(date +%Y%m%d_%H%M%S).txt"
echo "# Experiment Jobs - $(date)" > "$JOB_LOG"

# Submit function
submit() {
    local NAME="$1"
    local CONFIG="$2"
    local OVERRIDES="$3"
    
    SCRIPT=$(mktemp)
    cat > "$SCRIPT" << SLURM
#!/bin/bash
#SBATCH --job-name=${NAME}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --gres=gpu:${GPU}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=${LOG_DIR}/${NAME}_%j.out
#SBATCH --error=${LOG_DIR}/${NAME}_%j.err

echo "=== ${NAME} ===" 
echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURMD_NODENAME"
echo "Start: \$(date)"

cd ${PROJECT_DIR}
export PATH="\$HOME/miniforge3/bin:\$PATH"
source activate DPFM
export PYTHONPATH="\${PWD}/diffusion_policy:\$PYTHONPATH"

python dpfm/train.py --config-name=${CONFIG} ${OVERRIDES}

echo "End: \$(date)"
SLURM

    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY] $NAME: $CONFIG $OVERRIDES"
    else
        JOB_ID=$(sbatch "$SCRIPT" | awk '{print $4}')
        echo "$NAME: $JOB_ID" >> "$JOB_LOG"
        echo "✓ $NAME ($JOB_ID)"
    fi
    rm -f "$SCRIPT"
}

echo ""
echo "============================================"
echo "BASELINE EXPERIMENTS"
echo "============================================"

# --- DDPM Baselines ---
echo ">> DDPM Baselines"
submit "ddpm_unet_s42" "train_ddpm_unet_hybrid_pusht" \
    "exp_name=ddpm_unet training.seed=42"

# --- FM Baselines ---
echo ">> FM Baselines"
submit "fm_unet_s42" "train_fm_unet_hybrid_image_workspace" \
    "exp_name=fm_unet policy.num_inference_steps=4 training.seed=42"

submit "fm_trans_s42" "train_fm_transformer_hybrid_image_workspace" \
    "exp_name=fm_transformer policy.num_inference_steps=4 training.seed=42"

echo ""
echo "============================================"
echo "ABLATION A: INFERENCE STEPS (FM)"
echo "============================================"
for STEPS in 4 8 16; do
    submit "fm_steps${STEPS}" "train_fm_unet_hybrid_image_workspace" \
        "exp_name=fm_steps${STEPS} policy.num_inference_steps=${STEPS} training.seed=42 logging.group=ablation_steps"
done

echo ""
echo "============================================"
echo "ABLATION B: LR AND WARMUP (FM)"
echo "============================================"
for LR in "1e-4" "2e-4"; do
    for WARMUP in 500 1000; do
        NAME="fm_lr${LR}_w${WARMUP}"
        submit "$NAME" "train_fm_unet_hybrid_image_workspace" \
            "exp_name=${NAME} optimizer.lr=${LR} training.lr_warmup_steps=${WARMUP} policy.num_inference_steps=4 training.seed=42 logging.group=ablation_lr"
    done
done

echo ""
echo "============================================"
echo "ABLATION C: SEED SENSITIVITY (FM)"
echo "============================================"
for SEED in 42 43 44 45 46; do
    submit "fm_seed${SEED}" "train_fm_unet_hybrid_image_workspace" \
        "exp_name=fm_seed${SEED} policy.num_inference_steps=4 training.seed=${SEED} logging.group=ablation_seeds"
done

echo ""
echo "============================================"
echo "ABLATION D: DATA EFFICIENCY"
echo "============================================"
for DATA in 90 60 30; do
    # FM data ablation
    submit "fm_data${DATA}" "train_fm_unet_hybrid_image_workspace" \
        "exp_name=fm_data${DATA} task.dataset.max_train_episodes=${DATA} policy.num_inference_steps=4 training.seed=42 logging.group=ablation_data"
    
    # DDPM data ablation (for comparison)
    submit "ddpm_data${DATA}" "train_ddpm_unet_hybrid_pusht" \
        "exp_name=ddpm_data${DATA} task.dataset.max_train_episodes=${DATA} training.seed=42 logging.group=ablation_data"
done

echo ""
echo "============================================"
echo "SUMMARY"
echo "============================================"
if [[ "$DRY_RUN" == false ]]; then
    echo "Job log: $JOB_LOG"
    cat "$JOB_LOG"
fi
TOTAL=$(grep -c ":" "$JOB_LOG" 2>/dev/null || echo 0)
echo "Total jobs: $TOTAL"
