#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Module to create and run a TOI notebook from the template notebook"""

import argparse
import logging
import os
import re
import subprocess
from typing import Optional

import jupytext
import pkg_resources

from tess_atlas.utils import execute_ipynb
from .tess_atlas_version import __version__

logging.getLogger().setLevel(logging.INFO)


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
    version: Optional[str] = __version__,
    quickrun: Optional[bool] = False,
):
    """Creates a jupyter notebook for the TOI

    Args:
        toi_number: int
            The TOI Id number
        version: str
            The version of TESS Atlas being run
        quickrun: bool
            If True changes some constants to run the notebooks faster (useful for
            testing and debugging).

    Returns:
        notebook_filename: str
            The filepath of the generated notebook
    """
    notebook_filename = os.path.join(
        outdir, f"{version}/toi_{toi_number}.ipynb"
    )
    os.makedirs(os.path.dirname(notebook_filename), exist_ok=True)

    with open(get_template_filename(), "r") as f:
        txt = f.read()
        txt = txt.replace("{{{TOINUMBER}}}", f"{toi_number}")
        txt = txt.replace("{{{VERSIONNUMBER}}}", f"'{version}'")
        if quickrun:
            txt = re.sub(r"tune=[0-9]+", f"tune={5}", txt)
            txt = re.sub(r"draws=[0-9]+", f"draws={10}", txt)
            txt = re.sub(r"chains=[0-9]+", f"chains={1}", txt)
            txt = re.sub(r"cores=[0-9]+", f"cores={1}", txt)

    with open(notebook_filename, "w") as f:
        f.write(txt)

    return notebook_filename


def execute_toi_notebook(notebook_filename, version=__version__):
    """Executes the TOI notebook and git adds the notebook on a successful run.
    Prints an error on failure to run the notebook.

    Args:
        notebook_filename: str
            Filepath of the notebook
        version: str
            The string id of the TESS Atlas version for the run

    Returns:
        success: bool
            True if successful run of notebook
    """
    execution_successful = execute_ipynb(notebook_filename, version)
    if execution_successful:
        # subprocess.check_call(f"git add {notebook_filename} -f", shell=True)
        logging.info(f"Preprocessed {notebook_filename}")
    return execution_successful


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
    args = parser.parse_args()
    return args.toi_number, args.outdir


def main():
    toi_number, outdir = get_cli_args()
    notebook_filename = create_toi_notebook_from_template_notebook(
        toi_number, outdir
    )
    execute_toi_notebook(notebook_filename)


if __name__ == "__main__":
    main()
