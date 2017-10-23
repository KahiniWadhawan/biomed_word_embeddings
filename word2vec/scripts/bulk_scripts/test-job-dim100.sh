#!/bin/bash
#SBATCH --job-name test-job-dim100
#SBATCH --time 6:00:00
#SBATCH --nodes 1
#SBATCH --output test-job-dim100.out

echo "The job has begun."
sh kahini_dim100.sh
echo "Job completed."
