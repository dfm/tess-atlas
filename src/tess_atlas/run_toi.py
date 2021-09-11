#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Module to create and run a TOI notebook from the template notebook"""

import argparse
import os
import re
import time
from typing import Optional
import logging

import jupytext
import pkg_resources

from .utils import execute_ipynb
from .tess_atlas_version import __version__
from .logger import RUNNER_LOGGER_NAME, setup_logger


def get_template_filename():
    template_py_filename = pkg_resources.resource_filename(
        __name__, "template.py"
    )
    template_ipynb_filename = template_py_filename.replace(".py", ".ipynb")
    template_py_pointer = jupytext.read(template_py_filename, fmt="py:light")
    jupytext.write(template_py_pointer, template_ipynb_filename)
    return template_ipynb_filename


def create_toi_notebook_from_template_notebook(
    toi_number: int,
    outdir: Optional[str] = "notebooks",
    quickrun: Optional[bool] = False,
):
    """Creates a jupyter notebook for the TOI

    Args:
        toi_number: int
            The TOI Id number
        quickrun: bool
            If True changes some constants to run the notebooks faster (useful for
            testing and debugging).

    Returns:
        notebook_filename: str
            The filepath of the generated notebook
    """
    notebook_filename = os.path.join(
        outdir, f"{__version__}/toi_{toi_number}.ipynb"
    )
    os.makedirs(os.path.dirname(notebook_filename), exist_ok=True)

    with open(get_template_filename(), "r") as f:
        txt = f.read()
        txt = txt.replace("{{{TOINUMBER}}}", f"{toi_number}")
        txt = txt.replace("{{{VERSIONNUMBER}}}", f"'{__version__}'")
        if quickrun:
            txt = re.sub(r"tune=[0-9]+", f"tune={5}", txt)
            txt = re.sub(r"draws=[0-9]+", f"draws={10}", txt)
            txt = re.sub(r"chains=[0-9]+", f"chains={1}", txt)
            txt = re.sub(r"cores=[0-9]+", f"cores={1}", txt)

    with open(notebook_filename, "w") as f:
        f.write(txt)

    return notebook_filename


def execute_toi_notebook(notebook_filename):
    """Executes the TOI notebook and git adds the notebook on a successful run.
    Prints an error on failure to run the notebook.

    Args:
        notebook_filename: str
            Filepath of the notebook

    Returns:
        success: bool
            True if successful run of notebook
    """
    execution_successful = execute_ipynb(notebook_filename)
    return execution_successful


def get_cli_args():
    """Get the TOI number from the CLI and return it"""
    parser = argparse.ArgumentParser(prog="run_toi_in_pool")
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
    args = parser.parse_args()
    return args.toi_number, args.outdir, args.quickrun


def run_toi(toi_number, outdir, quickrun):
    t_start = time.time()
    notebook_filename = create_toi_notebook_from_template_notebook(
        toi_number, outdir, quickrun=quickrun
    )
    run_status = execute_toi_notebook(notebook_filename)
    t_end = time.time()
    run_duration = t_end - t_start
    record_run_stats(toi_number, run_status, run_duration, outdir)
    return run_status, run_duration


def record_run_stats(toi_number, run_status, run_duration, outdir):
    fname = os.path.join(outdir, __version__, "run_stats.csv")
    if not os.path.isfile(fname):
        open(fname, "w").write("toi,execution_complete,duration_in_s\n")
    open(fname, "a").write(f"{toi_number},{run_status},{run_duration}\n")


def main():
    toi_number, outdir, quickrun = get_cli_args()
    logger = setup_logger(
        RUNNER_LOGGER_NAME, outdir=os.path.join(outdir, __version__)
    )
    run_status, run_duration = run_toi(toi_number, outdir, quickrun)
    logger.info(
        f"TOI {toi_number} execution passed: {run_status} ({run_duration:.2f}s)"
    )


if __name__ == "__main__":
    main()
