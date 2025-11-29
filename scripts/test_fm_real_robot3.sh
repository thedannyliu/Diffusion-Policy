#!/bin/bash
#SBATCH --job-name=fm_real_test3
#SBATCH --output=logs/fm_real_test3_%j.out
#SBATCH --error=logs/fm_real_test3_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --account=gts-agarg35-ideas_l40s

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export WANDB_DIR=/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Diffusion-Policy-Flow-Matching/data/wandb
export WANDB_ENTITY=danny010324
export TQDM_MININTERVAL=5

source ~/.bashrc
conda activate DPFM

PROJECT_ROOT=/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Diffusion-Policy-Flow-Matching
cd $PROJECT_ROOT

echo "=== Testing FM Real Robot Training ==="
echo "Dataset: two_cameras_sphere (absolute path)"
echo "Starting at: $(date)"

# Quick test: 2 epochs only - use ABSOLUTE path
srun -u python dpfm/train.py \
    --config-name=train_fm_real_robot_workspace \
    task.dataset_path=${PROJECT_ROOT}/data/training/two_cameras_sphere \
    training.num_epochs=2 \
    training.debug=true \
    logging.mode=offline \
    exp_name=test_real_robot \
    hydra.run.dir='./data/outputs/test/${now:%H.%M.%S}_fm_real_robot_test'

echo "Test completed at: $(date)"
