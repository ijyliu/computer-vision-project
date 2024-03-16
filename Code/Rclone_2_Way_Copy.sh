#!/bin/bash

#SBATCH --job-name=Rclone_2_Way_Copy
#SBATCH --partition=lowmem
#SBATCH --output=Rclone_2_Way_Copy.out

echo "Making directory"

# Make directory if needed
mkdir -p ~/Box/"INFO 290T Project"

echo "Beginning Push"

# Push to remote/online version
rclone copy ~/Box/"INFO 290T Project" "Box:INFO 290T Project" --update

echo "Beginning Pull"

# Pull from remote/online version
rclone copy "Box:INFO 290T Project" ~/Box/"INFO 290T Project" --update
