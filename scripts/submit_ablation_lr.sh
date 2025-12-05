#!/bin/bash
# ============================================================================
# Submit LR Ablation Experiments (Phase 2)
# ============================================================================
# Submits training jobs for FM LR and warmup ablations
# Usage: bash scripts/submit_ablation_lr.sh [experiment_name]
#   - fm_lr1e-3: Learning rate 1e-3
#   - fm_lr5e-5: Learning rate 5e-5  
#   - fm_w1000: Warmup steps 1000
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
GPU="l40s:1"
# Exclude problematic nodes
EXCLUDE="atl1-1-03-004-31-0,atl1-1-01-010-35-0"

LOG_DIR="${PROJECT_DIR}/logs/experiments"
mkdir -p "$LOG_DIR"

submit_lr_ablation() {
    local NAME="$1"
    local LR="$2"
    local WARMUP="$3"
    
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
#SBATCH --exclude=${EXCLUDE}
#SBATCH --output=${LOG_DIR}/${NAME}_%j.out
#SBATCH --error=${LOG_DIR}/${NAME}_%j.err

echo "=== ${NAME} ===" 
echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURMD_NODENAME"
echo "Start: \$(date)"

cd ${PROJECT_DIR}
export PATH="\$HOME/miniforge3/bin:\$PATH"
source activate DPFM

# Show GPU info
nvidia-smi
echo "CUDA_VISIBLE_DEVICES: \$CUDA_VISIBLE_DEVICES"

export PYTHONPATH="\${PWD}/diffusion_policy:\$PYTHONPATH"

python dpfm/train.py --config-name=train_fm_unet_hybrid_image_workspace \
    exp_name=${NAME} \
    policy.num_inference_steps=4 \
    training.seed=42 \
    optimizer.lr=${LR} \
    training.lr_warmup_steps=${WARMUP}

echo "End: \$(date)"
SLURM

    JOB_ID=$(sbatch "$SCRIPT" | awk '{print $4}')
    echo "Submitted ${NAME}: Job ID ${JOB_ID}"
    rm -f "$SCRIPT"
}

EXP_NAME="$1"

case "$EXP_NAME" in
    "fm_lr1e-3")
        submit_lr_ablation "fm_lr1e-3" "1e-3" "500"
        ;;
    "fm_lr5e-5")
        submit_lr_ablation "fm_lr5e-5" "5e-5" "500"
        ;;
    "fm_w1000")
        submit_lr_ablation "fm_w1000" "1e-4" "1000"
        ;;
    "all")
        submit_lr_ablation "fm_lr1e-3" "1e-3" "500"
        submit_lr_ablation "fm_lr5e-5" "5e-5" "500"
        submit_lr_ablation "fm_w1000" "1e-4" "1000"
        ;;
    *)
        echo "Usage: $0 [fm_lr1e-3|fm_lr5e-5|fm_w1000|all]"
        echo ""
        echo "Available experiments:"
        echo "  fm_lr1e-3  - LR 1e-3 (10x baseline)"
        echo "  fm_lr5e-5  - LR 5e-5 (0.5x baseline)"
        echo "  fm_w1000   - Warmup 1000 steps (2x baseline)"
        echo "  all        - Submit all LR ablations"
        exit 1
        ;;
esac

echo ""
echo "Check status: squeue -u eliu354"
