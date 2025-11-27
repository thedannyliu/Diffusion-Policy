#!/bin/bash
#SBATCH --job-name=dpfm_test
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --partition=gpu-l40s
#SBATCH --qos=embers
#SBATCH --output=/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Diffusion-Policy-Flow-Matching/logs/dpfm_test_%j.out
#SBATCH --error=/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Diffusion-Policy-Flow-Matching/logs/dpfm_test_%j.err

# ============================================================
# Quick test script for Flow Matching implementation
# Runs with debug mode to verify everything works
# ============================================================

echo "=== DPFM Test Job ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Setup directories
PROJECT_DIR="/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Diffusion-Policy-Flow-Matching"
mkdir -p ${PROJECT_DIR}/logs

# Activate conda environment
source ~/.bashrc
conda activate DPFM

# Install missing packages - following the correct order for gym 0.21
echo ""
echo "=== Installing Dependencies ==="

# First: downgrade setuptools and wheel for gym 0.21 compatibility
pip install setuptools==65.5.0 wheel==0.38.4 --quiet

# Install gym 0.21 now that setuptools is compatible
pip install gym==0.21.0 --quiet

# Downgrade numpy for torch compatibility
pip install "numpy<2" --quiet

# Install other dependencies
pip install pandas scikit-image numba av pygame pymunk shapely opencv-python-headless zarr --quiet

echo "Installed packages:"
pip list | grep -E "gym|numpy|pandas|setuptools|wheel"

# Show environment info
echo ""
echo "=== Environment Info ==="
which python
python --version
nvidia-smi

# Set environment variables
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/diffusion_policy:${PYTHONPATH}"
export WANDB_MODE=offline

# Change to diffusion_policy directory (for data paths)
cd ${PROJECT_DIR}/diffusion_policy

# Test import
echo ""
echo "=== Testing Imports ==="
python -c "
import sys
sys.path.insert(0, '${PROJECT_DIR}')
sys.path.insert(0, '${PROJECT_DIR}/diffusion_policy')

print('Testing diffusion_policy imports...')
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
print('✓ diffusion_policy imports OK')

print('Testing dpfm imports...')
from dpfm.policy.flow_matching_unet_image_policy import FlowMatchingUnetImagePolicy
from dpfm.loss.flow_matching_loss import FlowMatchingLoss
from dpfm.sampler.euler_sampler import EulerSampler
print('✓ dpfm imports OK')

print('Testing torch...')
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
print('✓ All imports successful!')
"

# Run a quick training test with debug mode
echo ""
echo "=== Running Training Test (Debug Mode) ==="
export HYDRA_FULL_ERROR=1
python -m dpfm.train \
    --config-name=train_fm_unet_image_workspace \
    training.debug=True \
    training.device=cuda:0 \
    logging.mode=offline \
    hydra.run.dir=data/outputs/test_run

echo ""
echo "=== Job Completed ==="
echo "End time: $(date)"
