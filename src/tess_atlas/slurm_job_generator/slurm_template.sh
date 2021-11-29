#!/bin/bash
#
#SBATCH --job-name={{jobname}}
#SBATCH --output={{log_file}}
#
#SBATCH --ntasks=1
#SBATCH --time={{time}}
#SBATCH --mem={{mem}}
#SBATCH --cpus-per-task={{cpu_per_task}}
#SBATCH --array=0-{{array_end}}

module load {{module_loads}}
{{load_env}}

TOI_NUMBERS=({{toi_numbers}})

srun run_toi ${TOI_NUMBERS[$SLURM_ARRAY_TASK_ID]} --outdir {{outdir}} {{extra_jobargs}}
