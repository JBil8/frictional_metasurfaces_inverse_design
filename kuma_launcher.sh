#!/bin/bash
#SBATCH --job-name=nn_inverse_500k
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# --- Kuma Specifics ---
#SBATCH --partition=mig24gb  #l40s          
#SBATCH --gpus=1         
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4   
#SBATCH --mem=16G               
#SBATCH --time=01:00:00         
# --- Environment Setup ---
module purge
module load gcc                 # Load compiler
module load cuda         # Load CUDA

# Activate your virtual environment
source .venv/bin/activate

# 1. Generate Data (Only if it doesn't exist yet)
echo "Generating samples..."
python -m data.surface_generator

# 2. Train Model
echo "Starting Training on L40S..."
python main_inverse_design.py

# # 3. Evaluate Model
# echo "Evaluating Model..."
# python validation/validator.py