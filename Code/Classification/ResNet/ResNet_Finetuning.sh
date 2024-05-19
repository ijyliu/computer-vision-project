#!/bin/bash
#SBATCH --job-name=ResNet_Finetuning
#SBATCH --output=ResNet_Finetuning.out
#SBATCH --partition=jsteinhardt
#SBATCH --gres=gpu:A100:1

# Timing
# Reset the SECONDS variable
SECONDS=0

# Run script
python ResNet_Finetuning.py

# Calculate time in minutes
elapsed_minutes=$((SECONDS / 60))
# Print the time elapsed in minutes
echo "Time elapsed: $elapsed_minutes minute(s)."
