import os

COMMAND = "sacct -S {start} -E {end} -u {user} {extra_args} -o 'jobname%-40,cputimeraw,State,MaxRSS' --parsable2 > jobstats.txt"


def create_slurm_stats_file(start: str, end: str, user: str, store_mem: bool):
    """
    :param start: Date in YYYY-MM-DD format
    :param end: Date in YYYY-MM-DD format (must be greater than start)
    :param user: username
    :param store_mem: yes is mem data to be stored
    """
    extra_args = "" if store_mem else "-x"
    cmd = COMMAND.format(
        start=start, end=end, user=user, extra_args=extra_args
    )
    os.system(cmd)
