#!/bin/bash
#SBATCH --job-name test-job-dim200
#SBATCH --time 10:00:00
#SBATCH --nodes 1
#SBATCH --output test-job-dim200.out

echo "The job has begun."
sh kahini_dim200.sh
echo "Job completed."
