#!/bin/bash
#SBATCH --job-name=Fit_Random_Forest_Classifier_VGG
#SBATCH --output=Fit_Random_Forest_Classifier_VGG.out
#SBATCH --partition=epurdom
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

# Timing
# Reset the SECONDS variable
SECONDS=0

echo "Starting Job"

# Execute the notebook
jupyter nbconvert --to notebook --execute --inplace Fit_Random_Forest_Classifier_VGG.ipynb

echo "Completed Job"

# Calculate time in minutes
elapsed_minutes=$((SECONDS / 60))
# Print the time elapsed in minutes
echo "Time elapsed: $elapsed_minutes minute(s)."
