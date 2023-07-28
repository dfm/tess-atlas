#!/bin/bash
#
#SBATCH --job-name=toi_pe
#SBATCH --output=/home/runner/work/tess-atlas/tess-atlas/test_jobgen/log_pe/pe_%A_%a.log
#
#SBATCH --ntasks=1
#SBATCH --time=300:00
#SBATCH --mem=1500MB
#SBATCH --cpus-per-task=2
#SBATCH --tmp=500M

#SBATCH --array=0-14




module load module load 1

source /opt/hostedtoolcache/Python/3.8.17/x64/bin/activate

ARRAY_ARGS=(100 101 102 103 104 105 106 107 108 109 110 111 112 113 114)

srun run_toi ${ARRAY_ARGS[$SLURM_ARRAY_TASK_ID]} --outdir test_jobgen 