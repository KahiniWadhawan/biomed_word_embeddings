#!/bin/bash
#SBATCH --job-name test-job_text_win10
#SBATCH --time 10:00:00
#SBATCH --nodes 4
#SBATCH --output kahini_text_win10.out

echo "The job has begun."
sh kahini_text_win10.sh
echo "Job completed."
