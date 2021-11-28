import argparse
import os
import shutil
import sys
from typing import List, Optional

import jinja2
import pandas as pd

from ..data.exofop import get_toi_list

DIR = os.path.dirname(__file__)
TEMPLATE_FILE = "slurm_template.sh"


def load_template():
    template_loader = jinja2.FileSystemLoader(searchpath=DIR)
    template_env = jinja2.Environment(loader=template_loader)
    template = template_env.get_template(TEMPLATE_FILE)
    return template


def make_slurm_file(
    outdir: str,
    toi_numbers: List[int],
    module_loads: str,
    jobname: str,
    extra_jobargs: str,
    cpu_per_task: int,
    time: str,
    mem: str,
):
    template = load_template()
    path_to_python = shutil.which("python")
    path_to_env_activate = path_to_python.replace("python", "activate")
    file_contents = template.render(
        jobname=f"run_toi_{jobname}",
        time=time,
        outdir=os.path.abspath(outdir),
        log_file=os.path.join(outdir, f"toi_{jobname}_slurm_jobs.log"),
        module_loads=module_loads,
        total_num=str(len(toi_numbers) - 1),
        cpu_per_task=cpu_per_task,
        load_env=f"source {path_to_env_activate}",
        toi_numbers=" ".join([str(toi) for toi in toi_numbers]),
        extra_jobargs=extra_jobargs,
        mem=mem,
    )
    jobfile_name = os.path.join(outdir, f"slurm_{jobname}_job.sh")
    with open(jobfile_name, "w") as f:
        f.write(file_contents)
    return jobfile_name


def create_main_submitter(outdir, generation_fn, analysis_fn):
    lines = [
        "#!/bin/bash",
        f"GEN_ID=$(sbatch -p datamover --parsable {os.path.abspath(generation_fn)})",
        f"sbatch --dependency=aftercorr:$GEN_ID {os.path.abspath(analysis_fn)}",
        "squeue -u $USER -o '%.4u %.20j %.10A %.4C %.10E %R'",
        "",
    ]
    subfn = os.path.join(outdir, "submit.sh")
    with open(subfn, "w") as f:
        f.write("\n".join(lines))
    return subfn


def get_toi_numbers(toi_csv: str):
    df = pd.read_csv(toi_csv)
    return list(df.toi_numbers.values)


def make_toi_csv(fname: str, toi_numbers: Optional[List[int]] = []):
    if len(toi_numbers) == 0:
        toi_numbers = get_toi_list()
    toi_numbers = list(set([int(i) for i in toi_numbers]))
    data = pd.DataFrame(dict(toi_numbers=toi_numbers))
    data.to_csv(fname)


def get_cli_args(cli_data):
    parser = argparse.ArgumentParser(
        description="Create slurm job for analysing TOIs"
    )
    parser.add_argument(
        "--toi_csv",
        default=None,
        help="CSV with the toi numbers to analyse (csv needs a column with `toi_numbers`)",
    )
    parser.add_argument(
        "--outdir", help="outdir for jobs", default="notebooks"
    )
    parser.add_argument(
        "--module_loads",
        help="String containing all module loads in one line (each module separated by a space)",
    )
    parser.add_argument(
        "--submit",
        action="store_true",  # False by default
        help="Submit once files created",
    )
    parser.add_argument(
        "--toi_number",
        type=int,
        help="The TOI number to be analysed (e.g. 103). Cannot be passed with toi-csv",
        default=None,
    )
    args = parser.parse_args(cli_data)

    if args.toi_csv and args.toi_number is None:
        toi_numbers = get_toi_numbers(args.toi_csv)
    elif args.toi_csv is None and args.toi_number:
        toi_numbers = [args.toi_number]
    else:
        raise ValueError(
            f"You have provided TOI CSC: {args.toi_csv}, TOI NUMBER: {args.toi_number}."
            f"You need to provide one of the two (not both)."
        )

    return toi_numbers, args.outdir, args.module_loads, args.submit


def setup_jobs(
    toi_numbers: List[int], outdir: str, module_loads: str, submit: bool
) -> None:
    os.makedirs(outdir, exist_ok=True)

    generation_fn = make_slurm_file(
        outdir,
        toi_numbers,
        module_loads,
        extra_jobargs="--setup",
        cpu_per_task=1,
        time="20:00",
        jobname="generation",
        mem="500MB",
    )
    analysis_fn = make_slurm_file(
        outdir,
        toi_numbers,
        module_loads,
        extra_jobargs="",
        cpu_per_task=2,
        time="120:00",
        jobname="analysis",
        mem="1500MB",
    )

    submit_file = create_main_submitter(outdir, generation_fn, analysis_fn)

    if submit:
        os.system(f"bash {submit_file}")
    else:
        print(f"To run job:\n>>> bash {submit_file}")


def main():
    setup_jobs(*get_cli_args(sys.argv[1:]))


if __name__ == "__main__":
    main()
