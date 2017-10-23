#!/bin/bash
#SBATCH --job-name test-job_genia_300k
#SBATCH --time 10:00:00
#SBATCH --nodes 1
#SBATCH --output kahini_genia_300k.out

echo "The job has begun."
sh kahini_genia_300k.sh
echo "Job completed."
