#!/bin/bash
# Submit a single training job with proper node exclusion
# Usage: bash scripts/submit_single_job.sh <name> <config> "<overrides>"

set -e

NAME="$1"
CONFIG="$2"
OVERRIDES="$3"

PROJECT_DIR="/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Diffusion-Policy-Flow-Matching"
LOG_DIR="${PROJECT_DIR}/logs/experiments"
mkdir -p "$LOG_DIR"

# SLURM settings
ACCOUNT="gts-agarg35-ideas_l40s"
PARTITION="gpu-l40s"
TIME="20:00:00"
MEM="384G"
CPUS=8
GPU="l40s:1"
# Exclude nodes with known CUDA issues
EXCLUDE="atl1-1-03-004-31-0,atl1-1-01-010-35-0"

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
export PYTHONPATH="\${PWD}/diffusion_policy:\$PYTHONPATH"

python dpfm/train.py --config-name=${CONFIG} ${OVERRIDES}

echo "End: \$(date)"
SLURM

JOB_ID=$(sbatch "$SCRIPT" | awk '{print $4}')
echo "Submitted: $NAME (Job ID: $JOB_ID)"
rm -f "$SCRIPT"
