#!/bin/bash

GENERATION_FN=({{generation_fns}})
ANALYSIS_FN=({{analysis_fns}})
WEBGEN_FN=('{{webgen_fn}}')
ANLYS_IDS=()

for index in ${!GENERATION_FN[*]}; do
  echo "Submitting ${GENERATION_FN[$index]} ${ANALYSIS_FN[$index]}"
  GEN_ID=$(sbatch --parsable ${GENERATION_FN[$index]})
  ANLYS_ID=$(sbatch --parsable --dependency=aftercorr:$GEN_ID ${ANALYSIS_FN[$index]})
  ANLYS_IDS+=($ANLYS_ID)
done

JOBS=${ANLYS_IDS[@]}
JOBSTR=${JOBS// /:}

echo "Submitting ${WEBGEN_FN} after ${JOBSTR}"
sbatch --dependency=afterany:$JOBSTR $WEBGEN_FN

squeue -u $USER -o '%.4u %.20j %.10A %.4C %.10E %R'
