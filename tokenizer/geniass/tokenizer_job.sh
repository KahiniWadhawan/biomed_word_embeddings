#!/bin/bash
#SBATCH --job-name tokenizer-job
#SBATCH --time 24:00:00
#SBATCH --nodes 1
#SBATCH --output tokenizer-job.out

echo "The job has begun."
ml gcc/5.1.0 ; ml python/2.7.10
python tokenizer.py
echo "Job completed."
