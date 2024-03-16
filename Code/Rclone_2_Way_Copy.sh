#!/bin/bash

#SBATCH --job-name=Rclone_2_Way_Copy

# Suppress output
#SBATCH --output=/dev/null

# Push to remote/online version
rclone copy "~/Box/INFO 290T Project" Box:"INFO 290T Project" --update
# Pull from remote/online version
rclone copy Box:"INFO 290T Project" "~/Box/INFO 290T Project" --update
