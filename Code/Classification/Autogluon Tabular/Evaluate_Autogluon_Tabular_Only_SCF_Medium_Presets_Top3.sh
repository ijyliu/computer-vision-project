#!/bin/bash
#SBATCH --job-name=Evaluate_Autogluon_Tabular_Only_SCF_Medium_Presets_Top3
#SBATCH --output=Evaluate_Autogluon_Tabular_Only_SCF_Medium_Presets_Top3.out
#SBATCH --partition=jsteinhardt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A100:1

# Timing
# Reset the SECONDS variable
SECONDS=0

echo "Starting Job"

python "Evaluate_Autogluon_Tabular_Only_SCF_Medium_Presets_Top3.py"

echo "Completed Job"

# Calculate time in minutes
elapsed_minutes=$((SECONDS / 60))
# Print the time elapsed in minutes
echo "Time elapsed: $elapsed_minutes minute(s)."