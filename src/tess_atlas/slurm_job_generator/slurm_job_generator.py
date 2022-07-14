import os

import sys
from typing import List

from .slurm_cli import get_cli_args
from .file_generators import make_slurm_file, make_main_submitter
from .slurm_utils import get_unprocessed_toi_numbers
from tess_atlas.tess_atlas_version import __version__
from ..file_management import mkdir

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
    kwargs = dict(
        outdir=outdir,
        module_loads=module_loads,
        submit_dir=submit_dir,
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
            )
        )

    notebook_dir = os.path.join(outdir, __version__)
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
    else:
        print(f"To run job:\n>>> bash {submit_file}")


def main():
    setup_jobs(*get_cli_args(sys.argv[1:]))


if __name__ == "__main__":
    main()
