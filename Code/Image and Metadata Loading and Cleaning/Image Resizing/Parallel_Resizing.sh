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
echo "Train Blurred images:"
ls -1q ~/repo/computer-vision-project/Images/train/Blurred | wc -l
echo "Train Non-Blurred images:"
ls -1q ~/repo/computer-vision-project/Images/train/"No Blur" | wc -l
echo "Test Blurred images:"
ls -1q ~/repo/computer-vision-project/Images/test/Blurred | wc -l
echo "Test Non-Blurred images:"
ls -1q ~/repo/computer-vision-project/Images/test/"No Blur" | wc -l
