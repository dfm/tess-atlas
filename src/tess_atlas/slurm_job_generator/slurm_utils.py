import glob
import os
import re
import shutil
from typing import List, Optional

import pandas as pd

STATS_COMMAND = "sacct -S {start} -E {end} -u {user} {extra_args} -o 'jobname%-40,cputimeraw,State,MaxRSS' --parsable2 > jobstats.txt"


def create_slurm_stats_file(start: str, end: str, user: str, store_mem: bool):
    """This function creates a file called jobstats.txt in the current directory
    which contains the job stats for the given user between the given dates.

    Useful for debugging/checking job stats.

    The file is created by running the following command:
    sacct -S {start} -E {end} -u {user} {extra_args} -o \
    'jobname%-40,cputimeraw,State,MaxRSS' --parsable2 > jobstats.txt

    - cputimeraw is the total CPU time used by the job in seconds.
    - MaxRSS is the maximum resident set size of all tasks in the job.
    - State is the current state of the job (e.g. COMPLETED, FAILED, TIMEOUT).

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
