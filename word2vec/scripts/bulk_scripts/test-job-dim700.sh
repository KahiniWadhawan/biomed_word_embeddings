#!/bin/bash
#SBATCH --job-name test-job-dim700
#SBATCH --time 20:00:00
#SBATCH --nodes 1
#SBATCH --output test-job-dim700.out

echo "The job has begun."
sh kahini_dim700.sh
echo "Job completed."
