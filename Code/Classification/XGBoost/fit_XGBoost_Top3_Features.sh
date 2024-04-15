#!/bin/bash
#SBATCH --job-name=fit_XGBoost_Top3_Features
#SBATCH --output=fit_XGBoost_Top3_Features.out
#SBATCH --partition=epurdom
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

# Timing
# Reset the SECONDS variable
SECONDS=0

echo "Starting Job"

# Execute the notebook
jupyter nbconvert --to notebook --execute --inplace fit_XGBoost_Top3_Features.ipynb

echo "Completed Job"

# Calculate time in minutes
elapsed_minutes=$((SECONDS / 60))
# Print the time elapsed in minutes
echo "Time elapsed: $elapsed_minutes minute(s)."
