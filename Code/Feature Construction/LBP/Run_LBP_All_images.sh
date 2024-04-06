#!/bin/bash
#SBATCH --job-name=Run_LBP_All_images
#SBATCH --output=Run_LBP_All_images.out
#SBATCH --partition=jsteinhardt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

# Timing
# Reset the SECONDS variable
SECONDS=0

# Run script
python Run_LBP_All_images.py

# Calculate time in minutes
elapsed_minutes=$((SECONDS / 60))
# Print the time elapsed in minutes
echo "Time elapsed: $elapsed_minutes minute(s)."
