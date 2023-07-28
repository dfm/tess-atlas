#!/bin/bash
#
#SBATCH --job-name={{jobname}}
#SBATCH --output={{log_file}}
#
#SBATCH --ntasks=1
#SBATCH --time={{time}}
#SBATCH --mem={{mem}}
#SBATCH --cpus-per-task={{cpu_per_task}}
{% if tmp_mem!="" -%}      #SBATCH --tmp={{tmp_mem}}{% endif %}
{% if partition!="" -%}      #SBATCH --partition={{partition}}{% endif %}
{% if array_job=="True" -%}      #SBATCH --array=0-{{array_end}}{% endif %}
{% if email!="" -%}      #SBATCH --mail-user={{email}}{% endif %}
{% if email!="" -%}      #SBATCH --mail-type=ALL{% endif %}
{% if account!="" -%}      #SBATCH --account={{account}}{% endif %}

module load {{module_loads}}

{{load_env}}
{% if array_job=="True" %}
ARRAY_ARGS=({{array_args}})
{% endif %}
echo "Job tmp path: $JOBFS"
export THEANO_FLAGS="base_compiledir=$JOBFS/.theano_base,compiledir=$JOBFS/.theano_compile"
export IPYTHONDIR=$JOBFS/.ipython
{{command}}
