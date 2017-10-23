#!/bin/bash
#SBATCH --job-name test-job-subsample10_neg15
#SBATCH --time 16:00:00
#SBATCH --nodes 1
#SBATCH --output test-job-subsample10_neg15.out

echo "The job has begun."
sh kahini_subsample10_neg15.sh
echo "Job completed."
