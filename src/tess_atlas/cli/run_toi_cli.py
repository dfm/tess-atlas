import argparse
import os

from tess_atlas.file_management import TOI_DIR
from tess_atlas.logger import LOGGER_NAME, setup_logger, timestamp
from tess_atlas.notebook_controllers.controllers.toi_notebook_controller import (
    TOINotebookController,
)

PROG = "run_toi"


def __get_cli_args():
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Run TOI notebook",
        usage=f"{PROG} <toi_number> [--outdir <outdir>] [--quickrun] [--setup]",
    )
    default_outdir = os.path.join(os.getcwd(), "notebooks")
    parser.add_argument(
        "toi_number", type=int, help="The TOI number to be analysed (e.g. 103)"
    )
    parser.add_argument(
        "--outdir",
        default=default_outdir,
        type=str,
        help="The outdir to save notebooks (default: cwd/notebooks)",
    )
    parser.add_argument(
        "--quickrun",
        action="store_true",  # False by default
        help="Run with reduced sampler settings (useful for debugging)",
    )
    parser.add_argument(
        "--setup",
        action="store_true",  # False by default
        help="Setup data for run before executing notebook",
    )
    return parser.parse_args()


def main():
    args = __get_cli_args()
    logger = setup_logger(
        LOGGER_NAME,
        outdir=os.path.join(args.outdir, TOI_DIR.format(toi=args.toi_number)),
    )
    stmt = f"run_toi({args.toi_number}) {'quick' if args.quickrun else ''} {'setup' if args.setup else ''}"
    logger.info(stmt + f" [{timestamp()}]")
    success, runtime = TOINotebookController.run_toi(
        toi_number=args.toi_number,
        outdir=args.outdir,
        setup=args.setup,
        quickrun=args.quickrun,
    )
    job_str = "setup" if args.setup else "execution"
    logger.info(
        f"TOI {args.toi_number} {job_str} complete: {success} ({runtime:.2f}s) [{timestamp()}]"
    )
