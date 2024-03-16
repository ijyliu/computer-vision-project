#!/bin/bash
#SBATCH --job-name=Parallel_Resizing
#SBATCH --output=Parallel_Resizing.out
#SBATCH --partition=jsteinhardt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

# Timing
# Reset the SECONDS variable
SECONDS=0

# Run script
python Parallel_Resizing.py

# Calculate time in minutes
elapsed_minutes=$((SECONDS / 60))
# Print the time elapsed in minutes
echo "Time elapsed: $elapsed_minutes minute(s)."

# Count files
echo "Train images:"
ls -1q ~/Box/"INFO 290T Project/Intermediate Data/Resized Images"/train | wc -l
echo "Test images:"
ls -1q ~/Box/"INFO 290T Project/Intermediate Data/Resized Images"/test | wc -l
