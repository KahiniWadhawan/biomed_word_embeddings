#!/bin/bash
#SBATCH --job-name test-job-dim50
#SBATCH --time 6:00:00
#SBATCH --nodes 1
#SBATCH --output test-job-dim50.out

echo "The job has begun."
sh kahini_dim50.sh
echo "Job completed."
