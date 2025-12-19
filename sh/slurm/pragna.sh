#!/bin/bash
#SBATCH --job-name=pragna-train
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH -w soketlab-node005
#SBATCH --ntasks-per-node=8
#SBATCH --partition=gpu-long    # Replace with the correct partition name
#SBATCH --time=10-0:00:00
#SBATCH --output=logs/pragna-train_%x_%j.out
#SBATCH --error=logs/pragna-train_%x_%j.err

module load cuda/12.8
module load python/3.10
module load nccl

# Activate environment
source .venv/bin/activate # or conda activate your_env_name

srun litgpt pretrain --config config_hub/pretrain/pragna.yaml