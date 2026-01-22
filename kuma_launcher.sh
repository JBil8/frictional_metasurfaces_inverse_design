#!/bin/bash
#SBATCH --job-name=nn_inverse_500k
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# --- Kuma Specifics ---
#SBATCH --partition=l40s        # Request the L40S partition
#SBATCH --gpus=1                # Request 1 GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16      # 16 CPU cores to feed the GPU data
#SBATCH --mem=64G               # Request 64GB RAM (Safe for 500k samples)
#SBATCH --time=04:00:00         # 4 hours (adjust if needed)

# --- Environment Setup ---
module purge
module load gcc                 # Load compiler
module load cuda         # Load CUDA

# Activate your virtual environment
source ~/.venv/bin/activate

# 1. Generate Data (Only if it doesn't exist yet)
echo "Generating 500k samples..."
python -m data.surface_generator

# 2. Train Model
echo "Starting Training on L40S..."
python main_inverse_design.py