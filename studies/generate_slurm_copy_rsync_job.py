"""Make a slurm file to copy files from Ozstar to Necatr

Essentially we want to make the rsync command happen on the cluster, so we need to make a slurm file to do this.


The job will just need to run something like:



FROM OZSTAR:
    rsync -avPxH --no-g --chmod=Dg+s --progress -e 'ssh -i ~/.ssh/nectarkey.pem' {ozstar_dir} ec2-user@136.186.108.96:{dest_dir}

FROM NECATR:
    rsync -avPxH --no-g --chmod=Dg+s --progress avajpeyi@data-mover01.hpc.swin.edu.au:/fred/<project dir>/<somewhere>/ /mnt/storage/<local destination>
    rsync -avPxH --no-g --chmod=Dg+s --progress avajpeyi@data-mover01.hpc.swin.edu.au:/fred/<project dir>/<somewhere>/ /mnt/storage/<local destination>


"""

import argparse
import os
import sys

from tess_atlas.cli.make_slurm_job_cli import add_slurm_cli_args
from tess_atlas.slurm_job_generator.slurm_job_generator import make_slurm_file

COMMAND = (
    "rsync -avPxH --no-g --chmod=Dg+s --progress {extras} {src_dir} {dest_dir}"
)
SSH_KEY_EXTRA = " -e '{ssh}' "


def make_rsync_command(src, dst, keypath=""):
    extras = ""
    if keypath:
        extras += SSH_KEY_EXTRA.format(keypath=keypath)
    return COMMAND.format(extras=extras, src_dir=src, dest_dir=dst)


def parse_args(cli_args=None):
    if cli_args is None:
        cli_args = sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="Make a slurm file to copy files from Ozstar to Necatr"
    )
    parser.add_argument("-s", "--src", type=str, help="Source directory")
    parser.add_argument("-d", "--dst", type=str, help="Destination directory")
    parser.add_argument(
        "-k",
        "--ssh",
        type=str,
        default="",
        help="ssh extras (eg 'ssh -i ~/.ssh/nectarkey.pem ec2-user@136.186.108.96')",
    )
    parser = add_slurm_cli_args(parser)
    args = parser.parse_args(cli_args)
    return args


def main(cli_args=None):
    args = parse_args(cli_args)
    command = make_rsync_command(args.src, args.dst, keypath=args.keypath)
    make_slurm_file(
        command=command,
        jobname="rsync",
        outdir="out_rsync_job",
        module_loads=args.module_loads,
        cpu_per_task=1,
        time="02:00:00",
        mem="1G",
        submit_dir="out_rsync_job/submit",
        partition="datamover",
        email=args.email,
    )


if __name__ == "__main__":
    main()
