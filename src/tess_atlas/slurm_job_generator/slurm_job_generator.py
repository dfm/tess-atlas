import logging
import os
from typing import List

from tess_atlas.file_management import mkdir
from tess_atlas.logger import setup_logger

from .file_generators import make_main_submitter, make_slurm_file
from .slurm_toi_data_interface import get_unprocessed_toi_numbers

logger = setup_logger()

MAX_ARRAY_SIZE = 2048


def setup_jobs(
    toi_numbers: List[int],
    outdir: str,
    module_loads: str,
    submit: bool,
    clean: bool,
    email: str = "",
) -> None:
    """
    Generate slurm files for analysing TOIs
    - TOI number of parallel data generation jobs
    - TOI number of parallel analysis jobs
    - 1 job for generating webpages

    TODO: automation of getting new TOI list for analysis?
        To run on new TOIs
        1) run `update_tic_cache` (this will get the latest TOI list)
        2) run `make_slurm_jobs --submit` (this will generate the slurm files for the new TOIs + submit)
    TODO: automation of sending pages to Nectar to host?
    """

    logger.info(f"Generating slurm files for TESS ATLAS")

    initial_num, new_num = len(toi_numbers), len(toi_numbers)
    if not clean:
        # TODO: atm this only checks if the analysis outputs aare present.
        #       so, even if the generation job completed,
        #       this still re-runs the generation job, wasting resources
        # only run jobs for TOIs that have not been processed (ie no netcdf files)
        toi_numbers = get_unprocessed_toi_numbers(toi_numbers, outdir)
        new_num = len(toi_numbers)

    if new_num != initial_num:
        logger.info(
            f"TOIs to be processed: {new_num} (not analysing {initial_num - new_num})"
        )
    else:
        logger.info(f"_ALL_ TOIs to be processed: {initial_num}")

    submit_dir = mkdir(outdir, "submit")
    toi_batches = [
        toi_numbers[i : i + MAX_ARRAY_SIZE]
        for i in range(0, len(toi_numbers), MAX_ARRAY_SIZE)
    ]
    kwargs = dict(
        outdir=outdir,
        module_loads=module_loads,
        submit_dir=submit_dir,
        email=email,
    )

    generation_fns, analysis_fns = [], []

    for i, toi_batch in enumerate(toi_batches):
        cmd = "srun run_toi ${ARRAY_ARGS[$SLURM_ARRAY_TASK_ID]} "
        cmd += f"--outdir {outdir} "
        common_array_kwargs = dict(
            **kwargs,
            array_args=toi_batch,
            jobid=i,
            array_job=True,
        )
        generation_fns.append(
            make_slurm_file(
                **common_array_kwargs,
                cpu_per_task=1,
                time="20:00",
                jobname=f"gen",
                mem="1000MB",
                partition="datamover",
                command=f"{cmd} --setup",
            )
        )
        analysis_fns.append(
            make_slurm_file(
                **common_array_kwargs,
                cpu_per_task=2,
                time="300:00",
                jobname=f"pe",
                mem="1500MB",
                command=cmd,
                tmp_mem="500M",
            )
        )

    notebook_dir = os.path.join(outdir)
    web_fn = make_slurm_file(
        **kwargs,
        cpu_per_task=1,
        time="06:00:00",
        jobname=f"web",
        mem="64GB",
        command=f"make_webpages --webdir webpages --notebooks {notebook_dir} --add-api",
    )

    submit_file = make_main_submitter(
        generation_fns, analysis_fns, web_fn, submit_dir
    )

    if submit:
        os.system(f"bash {submit_file}")
        logger.info("All submitted!")
    else:
        logger.info(f"To run job:\n>>> bash {submit_file}")
