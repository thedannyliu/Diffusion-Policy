#!/bin/bash
#SBATCH --job-name=fm_cube
#SBATCH --output=logs/fm_cube_%j.out
#SBATCH --error=logs/fm_cube_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --account=gts-agarg35-ideas_l40s

# Train Flow Matching on two_cameras_cube dataset
# Usage: sbatch scripts/train_fm_cube.sh

export PYTHONUNBUFFERED=1
export WANDB_DIR=/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Diffusion-Policy-Flow-Matching/data/wandb
export WANDB_ENTITY=danny010324
export TQDM_MININTERVAL=5

source ~/.bashrc
conda activate DPFM

PROJECT_ROOT=/storage/home/hcoda1/9/eliu354/r-agarg35-0/projects/Diffusion-Policy-Flow-Matching
cd $PROJECT_ROOT

SEED=${1:-42}
STEPS=${2:-4}

echo "=== Training FM on two_cameras_cube ==="
echo "Seed: $SEED"
echo "Inference steps: $STEPS"
echo "Starting at: $(date)"

srun -u python dpfm/train.py \
    --config-name=train_fm_real_robot_workspace \
    task.dataset_path=${PROJECT_ROOT}/data/training/two_cameras_cube \
    task.name=real_robot_cube \
    training.seed=$SEED \
    policy.num_inference_steps=$STEPS \
    exp_name=fm_cube \
    logging.project=dpfm_real_robot \
    hydra.run.dir='./data/outputs/real_robot/${now:%Y.%m.%d}/${now:%H.%M.%S}_fm_cube_step${policy.num_inference_steps}_seed${training.seed}'

echo "Training completed at: $(date)"
