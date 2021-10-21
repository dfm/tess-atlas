import argparse
import os
import shutil
from typing import List

import jinja2
import pandas as pd

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
    time: str,
):
    template = load_template()
    path_to_python = shutil.which("python")
    path_to_env_activate = path_to_python.replace("python", "activate")
    file_contents = template.render(
        time=time,
        outdir=os.path.abspath(outdir),
        log_file=os.path.join(outdir, f"toi_{jobname}_slurm_jobs.log"),
        module_loads=module_loads,
        total_num=str(len(toi_numbers) - 1),
        load_env=f"source {path_to_env_activate}",
        toi_numbers=" ".join([str(toi) for toi in toi_numbers]),
        extra_jobargs=extra_jobargs,
    )
    jobfile_name = os.path.join(outdir, f"slurm_{jobname}_job.sh")
    with open(jobfile_name, "w") as f:
        f.write(file_contents)
    return jobfile_name


def get_toi_numbers(toi_csv: str):
    df = pd.read_csv(toi_csv)
    return list(df.toi_numbers.values)


def get_cli_args():
    parser = argparse.ArgumentParser(
        description="Create slurm job for analysing TOIs"
    )
    parser.add_argument(
        "--toi_csv",
        help="CSV with the toi numbers to analyse (csv needs a column with `toi_numbers`)",
    )
    parser.add_argument(
        "--outdir", help="outdir for jobs", default="notebooks"
    )
    parser.add_argument(
        "--module_loads",
        help="String containing all module loads in one line (each module separated by a space)",
    )
    args = parser.parse_args()
    return args.toi_csv, args.outdir, args.module_loads


def setup_jobs(toi_csv: str, outdir: str, module_loads: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    toi_numbers = get_toi_numbers(toi_csv)

    generation_fn = make_slurm_file(
        outdir,
        toi_numbers,
        module_loads,
        extra_jobargs="--setup",
        time="20:00",
        jobname="generation",
    )
    analysis_fn = make_slurm_file(
        outdir,
        toi_numbers,
        module_loads,
        extra_jobargs="",
        time="300:00",
        jobname="analysis",
    )

    print(
        f"""Jobfiles created, to run job:
    >>> sbatch -p datamover {generation_fn}
    (wait till above complete)
    >>> sbatch {analysis_fn}
    """
    )


def main():
    toi_csv, outdir, module_loads = get_cli_args()
    setup_jobs(toi_csv, outdir, module_loads)


if __name__ == "__main__":
    main()
