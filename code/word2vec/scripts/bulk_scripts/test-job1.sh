#!/bin/bash
#SBATCH --job-name test-job1
#SBATCH --time 24:00:00
#SBATCH --nodes 1
#SBATCH --output test-job1.out

echo "The job has begun."
sh kahini1.sh
echo "Job completed."
