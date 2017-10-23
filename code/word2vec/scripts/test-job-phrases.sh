#!/bin/bash
#SBATCH --job-name test-job-phrases
#SBATCH --time 24:00:00
#SBATCH --nodes 2
#SBATCH --output test-job-phrases.out

echo "The job has begun."
sh kahini-phrases.sh
echo "Job completed."
