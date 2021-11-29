#!/bin/bash

GENERATION_FN=({{generation_fns}})
ANALYSIS_FN=({{analysis_fns}})

function submit() {
    echo "Submitting ${1} ${2}"
    GEN_ID=$(sbatch -p datamover --parsable $1)
    sbatch --dependency=aftercorr:$GEN_ID $2
}

for index in ${!GENERATION_FN[*]}; do
  submit ${GENERATION_FN[$index]} ${ANALYSIS_FN[$index]}
done

squeue -u $USER -o '%.4u %.20j %.10A %.4C %.10E %R'
