#!/bin/bash
#
#SBATCH --job-name={{jobname}}
#SBATCH --output={{log_file}}
#
#SBATCH --ntasks=1
#SBATCH --time={{time}}
#SBATCH --mem={{mem}}
#SBATCH --cpus-per-task={{cpu_per_task}}
{% if array_job=="True" -%}      #SBATCH --array=0-{{array_end}}
{% endif %}
module load {{module_loads}}
{{load_env}}
{% if array_job=="True" %}
ARRAY_ARGS=({{array_args}})
{% endif %}
{{command}}
