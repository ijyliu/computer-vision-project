#!/bin/bash
#SBATCH --job-name=Run_HSV_All_Images
#SBATCH --output=Run_HSV_All_Images.out
#SBATCH --partition=jsteinhardt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

# Timing
# Reset the SECONDS variable
SECONDS=0

# Run script
python Run_HSV_All_Images.py

# Calculate time in minutes
elapsed_minutes=$((SECONDS / 60))
# Print the time elapsed in minutes
echo "Time elapsed: $elapsed_minutes minute(s)."
