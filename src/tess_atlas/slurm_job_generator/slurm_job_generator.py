import os

import sys
from typing import List

from .slurm_cli import get_cli_args
from .file_generators import make_slurm_file, make_main_submitter
from .slurm_utils import mkdir, get_unprocessed_toi_numbers

MAX_ARRAY_SIZE = 2048


def setup_jobs(
    toi_numbers: List[int],
    outdir: str,
    module_loads: str,
    submit: bool,
    clean: bool,
) -> None:

    if not clean:
        toi_numbers = get_unprocessed_toi_numbers(toi_numbers, outdir)

    submit_dir = mkdir(outdir, "submit")
    toi_batches = [
        toi_numbers[i : i + MAX_ARRAY_SIZE]
        for i in range(0, len(toi_numbers), MAX_ARRAY_SIZE)
    ]

    generation_fns, analysis_fns = [], []

    for i, toi_batch in enumerate(toi_batches):
        generation_fns.append(
            make_slurm_file(
                outdir=outdir,
                toi_numbers=toi_batch,
                module_loads=module_loads,
                extra_jobargs="--setup",
                cpu_per_task=1,
                time="20:00",
                jobname=f"gen",
                mem="1000MB",
                submit_dir=submit_dir,
                jobid=i,
            )
        )
        analysis_fns.append(
            make_slurm_file(
                outdir=outdir,
                toi_numbers=toi_batch,
                module_loads=module_loads,
                extra_jobargs="",
                cpu_per_task=2,
                time="120:00",
                jobname=f"pe",
                mem="1500MB",
                submit_dir=submit_dir,
                jobid=i,
            )
        )

    submit_file = make_main_submitter(generation_fns, analysis_fns, submit_dir)

    if submit:
        os.system(f"bash {submit_file}")
    else:
        print(f"To run job:\n>>> bash {submit_file}")


def main():
    setup_jobs(*get_cli_args(sys.argv[1:]))


if __name__ == "__main__":
    main()
