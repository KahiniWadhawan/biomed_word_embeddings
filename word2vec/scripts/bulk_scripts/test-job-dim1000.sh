#!/bin/bash
#SBATCH --job-name test-job-dim1000
#SBATCH --time 24:00:00
#SBATCH --nodes 1
#SBATCH --output test-job-dim1000.out

echo "The job has begun."
sh kahini_dim1000.sh
echo "Job completed."
