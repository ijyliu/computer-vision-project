#!/bin/bash
#SBATCH --job-name=CLIP_Relabeled_Test
#SBATCH --output=CLIP_Relabeled_Test.out
#SBATCH --partition=jsteinhardt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

# Timing
# Reset the SECONDS variable
SECONDS=0

# Run script
python CLIP_Relabeled_Test.py

# Calculate time in minutes
elapsed_minutes=$((SECONDS / 60))
# Print the time elapsed in minutes
echo "Time elapsed: $elapsed_minutes minute(s)."
