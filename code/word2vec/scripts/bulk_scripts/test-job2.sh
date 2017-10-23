#!/bin/bash
#SBATCH --job-name test-job2
#SBATCH --time 24:00:00
#SBATCH --nodes 1
#SBATCH --output test-job2.out

echo "The job has begun."
sh kahini2.sh
echo "Job completed."
