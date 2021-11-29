import os
import shutil
import sys
from typing import List

import jinja2
from .slurm_cli import get_cli_args

DIR = os.path.dirname(__file__)
SLURM_TEMPLATE = "slurm_template.sh"
SUBMIT_TEMPLATE = "submit_template.sh"

MAX_ARRAY_SIZE = 2048


def load_template(template_file: str):
    template_loader = jinja2.FileSystemLoader(searchpath=DIR)
    template_env = jinja2.Environment(loader=template_loader)
    template = template_env.get_template(template_file)
    return template


def get_python_source_command():
    path_to_python = shutil.which("python")
    path_to_env_activate = path_to_python.replace("python", "activate")
    return f"source {path_to_env_activate}"


def to_str_list(li):
    return " ".join([str(i) for i in li])


def make_slurm_file(
    outdir: str,
    toi_numbers: List[int],
    module_loads: str,
    jobname: str,
    jobid: int,
    extra_jobargs: str,
    cpu_per_task: int,
    time: str,
    mem: str,
    submit_dir: str,
):
    template = load_template(SLURM_TEMPLATE)
    log_dir = mkdir(outdir, f"log_{jobname}")
    os.makedirs(log_dir, exist_ok=True)
    file_contents = template.render(
        jobname=f"run_toi_{jobname}",
        time=time,
        outdir=os.path.abspath(outdir),
        log_file=os.path.join(log_dir, f"{jobname}_%A_%a.log"),
        module_loads=module_loads,
        array_end=str(len(toi_numbers) - 1),
        cpu_per_task=cpu_per_task,
        load_env=get_python_source_command(),
        toi_numbers=to_str_list(toi_numbers),
        extra_jobargs=extra_jobargs,
        mem=mem,
    )
    jobfile_name = os.path.join(submit_dir, f"slurm_{jobname}_{jobid}_job.sh")
    with open(jobfile_name, "w") as f:
        f.write(file_contents)
    return os.path.abspath(jobfile_name)


def mkdir(base, dirname):
    new_dir = os.path.join(base, dirname)
    os.makedirs(new_dir, exist_ok=True)
    return new_dir


def create_main_submitter(generation_fns, analysis_fns, submit_dir):
    template = load_template(SUBMIT_TEMPLATE)
    file_contents = template.render(
        generation_fns=to_str_list(generation_fns),
        analysis_fns=to_str_list(analysis_fns),
    )
    subfn = os.path.join(submit_dir, "submit.sh")
    with open(subfn, "w") as f:
        f.write(file_contents)
    return os.path.abspath(subfn)


def setup_jobs(
    toi_numbers: List[int], outdir: str, module_loads: str, submit: bool
) -> None:
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

    submit_file = create_main_submitter(
        generation_fns, analysis_fns, submit_dir
    )

    if submit:
        os.system(f"bash {submit_file}")
    else:
        print(f"To run job:\n>>> bash {submit_file}")


def main():
    setup_jobs(*get_cli_args(sys.argv[1:]))


if __name__ == "__main__":
    main()
