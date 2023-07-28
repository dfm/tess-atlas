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

ARRAY_ARGS=(115 116 117 118 119 120 121 122 123 124 125 126 127 128 129)

srun run_toi ${ARRAY_ARGS[$SLURM_ARRAY_TASK_ID]} --outdir test_jobgen 