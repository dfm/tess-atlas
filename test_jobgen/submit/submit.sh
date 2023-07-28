#!/bin/bash

GENERATION_FN=(/home/runner/work/tess-atlas/tess-atlas/test_jobgen/submit/slurm_gen_0_job.sh /home/runner/work/tess-atlas/tess-atlas/test_jobgen/submit/slurm_gen_1_job.sh)
ANALYSIS_FN=(/home/runner/work/tess-atlas/tess-atlas/test_jobgen/submit/slurm_pe_0_job.sh /home/runner/work/tess-atlas/tess-atlas/test_jobgen/submit/slurm_pe_1_job.sh)
WEBGEN_FN=('/home/runner/work/tess-atlas/tess-atlas/test_jobgen/submit/slurm_web_job.sh')
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