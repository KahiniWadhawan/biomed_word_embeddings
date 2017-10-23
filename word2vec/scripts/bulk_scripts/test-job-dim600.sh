#!/bin/bash
#SBATCH --job-name test-job-dim600
#SBATCH --time 16:00:00
#SBATCH --nodes 1
#SBATCH --output test-job-dim600.out

echo "The job has begun."
sh kahini_dim600.sh
echo "Job completed."
