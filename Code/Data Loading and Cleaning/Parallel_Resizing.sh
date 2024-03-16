#!/bin/bash
#SBATCH --job-name=Parallel_Resizing
#SBATCH --output=Parallel_Resizing.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --mem=4000

python Parallel_Resizing.py
