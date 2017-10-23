#!/bin/bash
#SBATCH --job-name test-job-dim150
#SBATCH --time 10:00:00
#SBATCH --nodes 1
#SBATCH --output test-job-dim150.out

echo "The job has begun."
sh kahini_dim150.sh
echo "Job completed."
