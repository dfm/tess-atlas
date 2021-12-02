import jinja2
from typing import List
import os

from .slurm_utils import mkdir, get_python_source_command, to_str_list

SLURM_TEMPLATE = "slurm_template.sh"
SUBMIT_TEMPLATE = "submit_template.sh"
DIR = os.path.dirname(__file__)


def load_template(template_file: str):
    template_loader = jinja2.FileSystemLoader(searchpath=DIR)
    template_env = jinja2.Environment(loader=template_loader)
    template = template_env.get_template(template_file)
    return template


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
    file_contents = template.render(
        jobname=f"run_toi_{jobname}",
        time=time,
        outdir=os.path.abspath(outdir),
        log_file=mkdir(log_dir, f"{jobname}_%A_%a.log"),
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


def make_main_submitter(generation_fns, analysis_fns, submit_dir):
    template = load_template(SUBMIT_TEMPLATE)
    file_contents = template.render(
        generation_fns=to_str_list(generation_fns),
        analysis_fns=to_str_list(analysis_fns),
    )
    subfn = os.path.join(submit_dir, "submit.sh")
    with open(subfn, "w") as f:
        f.write(file_contents)
    return os.path.abspath(subfn)
