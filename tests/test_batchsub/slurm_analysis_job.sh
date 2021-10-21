#!/bin/bash
#
#SBATCH --job-name=analysis
#SBATCH --output=log_analysis.log
#
#SBATCH --ntasks=1
#SBATCH --time=1:00
#SBATCH --mem-per-cpu=10MB
#
#SBATCH --array=0-9

module load git/2.18.0 gcc/9.2.0 openmpi/4.0.2 python/3.8.5

TOI_NUMBERS=(0 1 2 3 4 5 6 7 8 9)

srun echo ${TOI_NUMBERS[$SLURM_ARRAY_TASK_ID]} analysis
