#!/bin/bash
#SBATCH --job-name=ddpm_test
#SBATCH --output=logs/ddpm_test_%j.out
#SBATCH --error=logs/ddpm_test_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --account=gts-agarg35-ideas_l40s

# Quick test: DDPM baseline
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export WANDB_DIR=/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Diffusion-Policy-Flow-Matching/data/wandb
export WANDB_ENTITY=danny010324
export TQDM_MININTERVAL=5

source ~/.bashrc
conda activate DPFM

PROJECT_ROOT=/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Diffusion-Policy-Flow-Matching
cd $PROJECT_ROOT/diffusion_policy

echo "=== Testing DDPM Baseline ==="
echo "Starting at: $(date)"

# Need to override shape_meta for two_cameras dataset
srun -u python train.py \
    --config-name=train_diffusion_unet_real_image_workspace \
    task.dataset_path=${PROJECT_ROOT}/data/training/two_cameras_sphere \
    'task.shape_meta.obs.camera_0.shape=[3,240,320]' \
    'task.shape_meta.obs.camera_0.type=rgb' \
    'task.shape_meta.obs.camera_1.shape=[3,240,320]' \
    'task.shape_meta.obs.camera_1.type=rgb' \
    'task.shape_meta.obs.robot_eef_pose.shape=[6]' \
    'task.shape_meta.obs.robot_eef_pose.type=low_dim' \
    'task.shape_meta.action.shape=[6]' \
    training.num_epochs=2 \
    training.debug=true \
    logging.mode=offline \
    hydra.run.dir='${PROJECT_ROOT}/data/outputs/test/${now:%H.%M.%S}_ddpm_test'

echo "Test completed at: $(date)"
