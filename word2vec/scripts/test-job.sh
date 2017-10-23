#!/bin/bash
#SBATCH --job-name test-job
#SBATCH --time 24:00:00
#SBATCH --nodes 4
#SBATCH --output test-job.out

echo "The job has begun."
sh kahini.sh
echo "Job completed."
