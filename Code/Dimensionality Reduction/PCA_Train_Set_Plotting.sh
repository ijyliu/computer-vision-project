#!/bin/bash
#SBATCH --job-name=PCA_Train_Set_Plotting
#SBATCH --output=PCA_Train_Set_Plotting.out
#SBATCH --partition=jsteinhardt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# Timing
# Reset the SECONDS variable
SECONDS=0

echo "Starting Job"

# Execute the notebook
jupyter nbconvert --to notebook --execute --inplace PCA_Train_Set_Plotting.ipynb

echo "Completed Job"

# Calculate time in minutes
elapsed_minutes=$((SECONDS / 60))
# Print the time elapsed in minutes
echo "Time elapsed: $elapsed_minutes minute(s)."
