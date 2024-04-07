#!/bin/bash
#SBATCH --job-name=Fit_Logistic_Regression_Classifier
#SBATCH --output=Fit_Logistic_Regression_Classifier.out
#SBATCH --partition=jsteinhardt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

# Timing
# Reset the SECONDS variable
SECONDS=0

# Run script
python Fit_Logistic_Regression_Classifier.py

# Calculate time in minutes
elapsed_minutes=$((SECONDS / 60))
# Print the time elapsed in minutes
echo "Time elapsed: $elapsed_minutes minute(s)."
