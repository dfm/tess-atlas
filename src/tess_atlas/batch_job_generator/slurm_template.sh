#!/bin/bash
#
#SBATCH --job-name=run_tois
#SBATCH --output={{{LOG FILE}}}
#
#SBATCH --ntasks=1
#SBATCH --time=300:00
#SBATCH --mem-per-cpu=500MB
#
#SBATCH --array=0-{{{TOTAL NUM}}}

{{{MODULE LOADS}}}
{{{LOAD ENV}}}


TOI_NUMBERS=({{{TOI NUMBERS}}})

srun run_toi ${TOI_NUMBERS[$SLURM_ARRAY_TASK_ID]} --outdir {{{OUTDIR}}}
