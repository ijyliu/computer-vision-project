#!/bin/bash
#SBATCH --job-name=VGG_SCF
#SBATCH --output=VGG_SCF.out
#SBATCH --partition=yss
#SBATCH --gres=gpu:A100:1

# Timing
# Reset the SECONDS variable
SECONDS=0

# Run script
python VGG_SCF.py

# Calculate time in minutes
elapsed_minutes=$((SECONDS / 60))
# Print the time elapsed in minutes
echo "Time elapsed: $elapsed_minutes minute(s)."
