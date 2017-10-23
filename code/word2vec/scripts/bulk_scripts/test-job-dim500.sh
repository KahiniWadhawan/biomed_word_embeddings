#!/bin/bash
#SBATCH --job-name test-job-dim500
#SBATCH --time 16:00:00
#SBATCH --nodes 1
#SBATCH --output test-job-dim500.out

echo "The job has begun."
sh kahini_dim500.sh
echo "Job completed."
