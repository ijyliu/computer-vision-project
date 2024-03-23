#!/bin/bash
#SBATCH --job-name=CLIP_Relabeled_Test_GPU
#SBATCH --output=CLIP_Relabeled_Test_GPU.out
#SBATCH --partition=jsteinhardt
#SBATCH --gres=gpu:A100:1

# Timing
# Reset the SECONDS variable
SECONDS=0

# Run script
python CLIP_Relabeled_Test_GPU.py

# Calculate time in minutes
elapsed_minutes=$((SECONDS / 60))
# Print the time elapsed in minutes
echo "Time elapsed: $elapsed_minutes minute(s)."
