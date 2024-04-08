#!/bin/bash
#SBATCH --job-name=Fit_SVM_Classifier_All_Features_PCA
#SBATCH --output=Fit_SVM_Classifier_All_Features_PCA.out
#SBATCH --partition=jsteinhardt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

# Timing
# Reset the SECONDS variable
SECONDS=0

echo "Starting Job"

# Execute the notebook
jupyter nbconvert --to notebook --execute --inplace Fit_SVM_Classifier_All_Features_PCA.ipynb

echo "Completed Job"

# Calculate time in minutes
elapsed_minutes=$((SECONDS / 60))
# Print the time elapsed in minutes
echo "Time elapsed: $elapsed_minutes minute(s)."
