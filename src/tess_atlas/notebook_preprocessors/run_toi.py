#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Module to create and run a TOI notebook from the template notebook"""

import argparse
import logging
import os
import re
import time
from typing import Optional, Tuple

from ..data.tic_entry import TICEntry
from ..tess_atlas_version import __version__
from ..utils import RUNNER_LOGGER_NAME, setup_logger
from .notebook_executor import execute_ipynb
from .toi_notebook_generator import create_toi_notebook_from_template_notebook


def get_cli_args():
    """Get the TOI number from the CLI and return it"""
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
    args = parser.parse_args()
    return args.toi_number, args.outdir, args.quickrun, args.setup


def run_toi(
    toi_number: int,
    outdir: str,
    quickrun: Optional[bool] = False,
    setup: Optional[bool] = False,
) -> Tuple[bool, float]:
    """Creates+preprocesses TOI notebook and records the executions' stats.

    Args:
        toi_number: int
            The TOI Id number
        quickrun: bool
            If True changes sampler settings to run the notebooks faster
            (useful for testing/debugging -- produces non-scientific results)
        outdir: str
            Base outdir for TOI. Notebook will be saved at
            {outdir}/{tess_atlas_version}/toi_{toi_number}.ipynb}
        setup: bool
            If true creates notebook + downloads data needed for analysis
            but does not execute notebook

    Returns:
        execution_successful: bool
            True if successful run of notebook
        run_duration: float
            Time of analysis (in seconds)
    """
    t0 = time.time()
    notebook_filename = create_toi_notebook_from_template_notebook(
        toi_number, outdir, quickrun=quickrun, setup=setup
    )
    execution_successful = True
    if not setup:
        execution_successful = execute_ipynb(notebook_filename)
        record_run_stats(
            toi_number, execution_successful, time.time() - t0, outdir
        )
    return execution_successful, time.time() - t0


def record_run_stats(
    toi_number: int,
    execution_successful: bool,
    run_duration: float,
    outdir: str,
):
    """Creates/Appends to a CSV the runtime and status of the TOI analysis."""
    fname = os.path.join(outdir, __version__, "run_stats.csv")
    if not os.path.isfile(fname):
        open(fname, "w").write("toi,execution_complete,duration_in_s\n")
    open(fname, "a").write(
        f"{toi_number},{execution_successful},{run_duration}\n"
    )


def main():
    toi_number, outdir, quickrun, setup = get_cli_args()
    logger = setup_logger(
        RUNNER_LOGGER_NAME, outdir=os.path.join(outdir, __version__)
    )
    logger.info(
        f"run_toi({toi_number}) {'quick' if quickrun else ''} {'setup' if setup else ''}"
    )
    success, run_duration = run_toi(toi_number, outdir, quickrun, setup)
    job_str = "setup" if setup else "execution"
    logger.info(
        f"TOI {toi_number} {job_str} complete: {success} ({run_duration:.2f}s)"
    )


if __name__ == "__main__":
    main()
