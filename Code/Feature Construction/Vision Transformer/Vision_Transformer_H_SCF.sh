#!/bin/bash
#SBATCH --job-name=Vision_Transformer_H_SCF
#SBATCH --output=Vision_Transformer_H_SCF.out
#SBATCH --partition=jsteinhardt
#SBATCH --gres=gpu:A100:1

# Timing
# Reset the SECONDS variable
SECONDS=0

# Run script
python Vision_Transformer_H_SCF.py

# Calculate time in minutes
elapsed_minutes=$((SECONDS / 60))
# Print the time elapsed in minutes
echo "Time elapsed: $elapsed_minutes minute(s)."
