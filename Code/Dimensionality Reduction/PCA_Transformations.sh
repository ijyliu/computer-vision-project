#!/bin/bash
#SBATCH --job-name=PCA_Transformations
#SBATCH --output=PCA_Transformations.out
#SBATCH --partition=jsteinhardt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# Timing
# Reset the SECONDS variable
SECONDS=0

echo "Starting Job"

python PCA_Transformations.py

echo "Completed Job"

# Calculate time in minutes
elapsed_minutes=$((SECONDS / 60))
# Print the time elapsed in minutes
echo "Time elapsed: $elapsed_minutes minute(s)."
