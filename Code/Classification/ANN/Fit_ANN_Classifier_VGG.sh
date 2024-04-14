#!/bin/bash
#SBATCH --job-name=Fit_ANN_Classifier_VGG
#SBATCH --output=Fit_ANN_Classifier_VGG.out
#SBATCH --partition=jsteinhardt
#SBATCH --gres=gpu:A100:1

# Timing
# Reset the SECONDS variable
SECONDS=0

# Run script
python Fit_ANN_Classifier_VGG.py

# Calculate time in minutes
elapsed_minutes=$((SECONDS / 60))
# Print the time elapsed in minutes
echo "Time elapsed: $elapsed_minutes minute(s)."
