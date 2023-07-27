import argparse
import os

from tess_atlas.cli.make_slurm_job_cli import add_slurm_cli_args
from tess_atlas.data.exofop.exofop_database import ExofopDatabase
from tess_atlas.logger import setup_logger
from tess_atlas.slurm_job_generator import make_slurm_file

logger = setup_logger()


PROG = "update_tic_cache"


def get_cli_args():
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Update TIC cache from ExoFOP database. "
            "NOTE: Lightkurve is queried for each TIC ID to "
            "verify that the correct data is available."
        ),
        usage=f"{PROG} [--clean] [--slurm]",
    )
    parser.add_argument(
        "--clean",
        action="store_true",  # False by default
        help="Update cache from scratch",
    )
    parser.add_argument(
        "--slurm",
        action="store_true",  # false by default
        help="true if you want to make a slurm job file",
    )
    parser = add_slurm_cli_args(parser)
    args = parser.parse_args()
    return args


def main():
    args = get_cli_args()
    if args.slurm:
        command = (
            "update_tic_cache"
            if not args.clean
            else "update_tic_cache --clean"
        )
        outdir = os.path.join(os.getcwd(), "out_tic_cache_update")
        os.makedirs(outdir, exist_ok=True)
        fn = make_slurm_file(
            outdir=outdir,
            module_loads=args.module_loads,
            jobname="update_tic_cache",
            cpu_per_task=1,
            time="00:30:00",
            mem="1000MB",
            partition="datamover",
            submit_dir=outdir,
            command=command,
        )
        logger.info(f"To run job:\n>>> sbatch {fn}")
    else:
        logger.info(f"UPDATING TIC CACHE... clean={args.clean}")
        db = ExofopDatabase(update=True, clean=args.clean)
        db.plot()
        logger.info("UPDATE COMPLETE!!")
