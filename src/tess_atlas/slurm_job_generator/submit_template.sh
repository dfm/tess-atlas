#!/bin/bash

GENERATION_FN=({{generation_fns}})
ANALYSIS_FN=({{analysis_fns}})

for index in ${!GENERATION_FN[*]}; do
  G_FN=$($GENERATION_FN[$index])
  A_FN=$($ANALYSIS_FN[$index])
  GEN_ID=$(sbatch -p datamover --parsable $G_FN)
  sbatch --dependency=aftercorr:$GEN_ID $A_FN
done

squeue -u $USER -o '%.4u %.20j %.10A %.4C %.10E %R'
