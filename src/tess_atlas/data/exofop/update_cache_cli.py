import argparse
import os

from tess_atlas.slurm_job_generator import add_slurm_cli_args, make_slurm_file

from .tic_database import TICDatabase


def get_cli_args():
    """Get CLI arguments"""
    parser = argparse.ArgumentParser(prog="update_tic_cache")
    parser.add_argument(
        "--clean",
        action="store_true",  # False by default
        help="Update cache from scratch",
    )
    parser = add_slurm_cli_args(parser)
    parser.add_argument(
        "--slurm",
        action="store_true",  # false by default
        help="true if you want to make a slurm job file",
    )
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
        print(f"To run job:\n>>> sbatch {fn}")
    else:
        db = TICDatabase(update=True, clean=args.clean)
        db.plot_caches()
        print("UPDATE COMPLETE!!")
