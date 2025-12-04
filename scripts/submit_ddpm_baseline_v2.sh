#!/bin/bash
# Submit DDPM UNet baseline job (excluding bad nodes)
# Usage: bash scripts/submit_ddpm_baseline_v2.sh

set -e
cd /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Diffusion-Policy-Flow-Matching

LOG_DIR="logs/experiments"
mkdir -p "$LOG_DIR"

# DDPM UNet Baseline - exclude problematic node
JOB_SCRIPT=$(mktemp)
cat > "$JOB_SCRIPT" << 'JOBSCRIPT'
#!/bin/bash
#SBATCH --job-name=ddpm_unet_s42
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=384G
#SBATCH --time=20:00:00
#SBATCH --exclude=atl1-1-03-004-31-0,atl1-1-03-004-29-0
#SBATCH --output=logs/experiments/ddpm_unet_s42_%j.out
#SBATCH --error=logs/experiments/ddpm_unet_s42_%j.err

echo "=========================================="
echo "Experiment: ddpm_unet_s42"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"
echo "=========================================="

cd /storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Diffusion-Policy-Flow-Matching
export PATH="$HOME/miniforge3/bin:$PATH"
source activate DPFM
export PYTHONPATH="${PWD}/diffusion_policy:$PYTHONPATH"

python dpfm/train.py \
    --config-name=train_ddpm_unet_hybrid_pusht \
    exp_name=ddpm_unet \
    training.seed=42

echo "=========================================="
echo "Completed: $(date)"
echo "=========================================="
JOBSCRIPT

JOB_ID=$(sbatch "$JOB_SCRIPT" | awk '{print $4}')
echo "Submitted ddpm_unet_s42: Job ID $JOB_ID"
rm "$JOB_SCRIPT"
