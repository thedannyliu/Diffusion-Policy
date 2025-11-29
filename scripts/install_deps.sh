#!/bin/bash
#SBATCH --job-name=install_deps
#SBATCH --output=logs/install_deps_%j.out
#SBATCH --error=logs/install_deps_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --account=gts-agarg35-ideas_l40s

source ~/.bashrc
conda activate DPFM

echo "Installing missing dependencies..."
pip install threadpoolctl filelock imagecodecs av numcodecs

echo "Done!"
