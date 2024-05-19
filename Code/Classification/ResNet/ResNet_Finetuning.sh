#!/bin/bash
#SBATCH --job-name=ResNet_Finetuning
#SBATCH --output=ResNet_Finetuning.out
#SBATCH --partition=jsteinhardt
#SBATCH --gres=gpu:A100:1

# Timing
# Reset the SECONDS variable
SECONDS=0

# Execute the notebook
jupyter nbconvert --to notebook --execute --inplace ResNet_Finetuning.ipynb

# Calculate time in minutes
elapsed_minutes=$((SECONDS / 60))
# Print the time elapsed in minutes
echo "Time elapsed: $elapsed_minutes minute(s)."
