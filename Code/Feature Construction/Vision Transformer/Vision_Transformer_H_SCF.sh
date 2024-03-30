#!/bin/bash
#SBATCH --job-name=Vision_Transformer_H_SCF
#SBATCH --output=Vision_Transformer_H_SCF.out
#SBATCH --partition=jsteinhardt
#SBATCH --gres=gpu:A100:1

echo "Starting ViT Embedding Creation"

python Vision_Transformer_H_SCF.py

echo "Completed ViT Embedding Creation"
