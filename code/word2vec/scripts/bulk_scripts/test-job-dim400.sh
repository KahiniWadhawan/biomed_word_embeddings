#!/bin/bash
#SBATCH --job-name test-job-dim400
#SBATCH --time 15:00:00
#SBATCH --nodes 1
#SBATCH --output test-job-dim400.out

echo "The job has begun."
sh kahini_dim400.sh
echo "Job completed."
