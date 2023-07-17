import argparse
import os

from ...file_management import TOI_DIR
from ...logger import LOGGER_NAME, setup_logger
from ..controllers import TOINotebookController

__all__ = ["cli_run_toi"]


def cli_run_toi():
    args = __get_cli_args()
    logger = setup_logger(
        LOGGER_NAME,
        outdir=os.path.join(args.outdir, TOI_DIR.format(toi=args.toi_number)),
    )
    logger.info(
        f"run_toi({args.toi_number}) {'quick' if args.quickrun else ''} {'setup' if args.setup else ''}"
    )
    nb_controller = TOINotebookController.from_toi_number(
        args.toi_number, args.outdir
    )
    nb_controller.generate(setup=args.setup, quickrun=args.quickrun)
    success = nb_controller.execute()
    run_duration = nb_controller.execution_time
    job_str = "setup" if args.setup else "execution"
    logger.info(
        f"TOI {args.toi_number} {job_str} complete: {success} ({run_duration:.2f}s)"
    )


def __get_cli_args():
    parser = argparse.ArgumentParser(prog="run_toi")
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
