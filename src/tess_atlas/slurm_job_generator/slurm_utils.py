import os
import shutil
import glob
import re
import pandas as pd
from typing import List

STATS_COMMAND = "sacct -S {start} -E {end} -u {user} {extra_args} -o 'jobname%-40,cputimeraw,State,MaxRSS' --parsable2 > jobstats.txt"


def create_slurm_stats_file(start: str, end: str, user: str, store_mem: bool):
    """
    :param start: Date in YYYY-MM-DD format
    :param end: Date in YYYY-MM-DD format (must be greater than start)
    :param user: username
    :param store_mem: yes is mem data to be stored
    """
    extra_args = "" if store_mem else "-x"
    cmd = STATS_COMMAND.format(
        start=start, end=end, user=user, extra_args=extra_args
    )
    os.system(cmd)


def get_python_source_command():
    path_to_python = shutil.which("python")
    path_to_env_activate = path_to_python.replace("python", "activate")
    return f"source {path_to_env_activate}"


def to_str_list(li):
    return " ".join([str(i) for i in li])


def mkdir(base, name):
    newpth = os.path.join(base, name)
    os.makedirs(os.path.dirname(newpth), exist_ok=True)
    return newpth


def get_completed_toi_pe_results_paths(outdir: str):
    search_path = os.path.join(outdir, "*/toi_*_files/*.netcdf")
    netcdf_files = glob.glob(search_path)
    regex = "toi_(\d+)_files"
    tois = [int(re.search(regex, f).group(1)) for f in netcdf_files]
    return pd.DataFrame(dict(TOI=tois, path=netcdf_files))


def get_unprocessed_toi_numbers(toi_numbers: List, outdir: str):
    processed_tois = set(get_completed_toi_pe_results_paths(outdir).TOI.values)
    tois = set(toi_numbers)
    return list(tois.difference(processed_tois))
