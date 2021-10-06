import argparse
import os
import shutil
from typing import List

import pandas as pd

TEMPLATE_FILE = os.path.join(os.path.dirname(__file__), "slurm_template.sh")


def make_slurm_file(outdir: str, toi_numbers: List[int], module_loads: str):
    with open(TEMPLATE_FILE, "r") as f:
        file_contents = f.read()
    outdir = os.path.abspath(outdir)
    logfile_name = os.path.join(outdir, "toi_slurm_jobs.log")
    jobfile_name = os.path.join(outdir, "slurm_job.sh")
    path_to_python = shutil.which("python")
    path_to_env_activate = path_to_python.replace("python", "activate")
    file_contents = file_contents.replace(
        "{{{TOTAL NUM}}}", str(len(toi_numbers) - 1)
    )
    file_contents = file_contents.replace("{{{MODULE LOADS}}}", module_loads)
    file_contents = file_contents.replace("{{{OUTDIR}}}", outdir)
    file_contents = file_contents.replace(
        "{{{LOAD ENV}}}", f"source {path_to_env_activate}"
    )
    file_contents = file_contents.replace("{{{LOG FILE}}}", logfile_name)
    toi_str = " ".join([str(toi) for toi in toi_numbers])
    file_contents = file_contents.replace("{{{TOI NUMBERS}}}", toi_str)
    with open(jobfile_name, "w") as f:
        f.write(file_contents)
    print(f"Jobfile created, to run job: \nsbatch {jobfile_name}")


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


def main():
    toi_csv, outdir, module_loads = get_cli_args()
    os.makedirs(outdir, exist_ok=True)
    toi_numbers = get_toi_numbers(toi_csv)
    make_slurm_file(outdir, toi_numbers, module_loads)


if __name__ == "__main__":
    main()
