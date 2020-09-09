#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Module to create and run a TOI notebook from the template notebook"""

from __future__ import division, print_function

import os
import re
import subprocess
import sys
from typing import Optional

import nbformat
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor

from tess_atlas.tess_atlas_version import __version__

TEMPLATE = "template.ipynb"


def create_toi_notebook_from_template_notebook(
        toi_number: int,
        version: Optional[str] = __version__,
        quickrun: Optional[bool] = False
):
    """Creates a jupyter notebook for the TOI

    Args:
        toi_number: int
            The TOI Id number
        version: str
            The version of TESS Atlas being run
        quickrun: bool
            If True changes some constants to run the notebooks faster (useful for
            testing and debugging). CURRENTLY UNIMPLEMENTED.

    Returns:
        notebook_filename: str
            The filepath of the generated notebook
    """
    notebook_filename = f"notebooks/{version}/toi-{toi_number}.ipynb"
    os.makedirs(os.path.dirname(notebook_filename), exist_ok=True)

    with open(TEMPLATE, "r") as f:
        txt = f.read()
        txt = txt.replace("{{{TOINUMBER}}}", f"{toi_number}")
        txt = txt.replace("{{{VERSIONNUMBER}}}", f"'{version}'")
        txt = re.sub(r"toi_num = [0-9]+", f"toi_num = {toi_number}", txt)
        if quickrun:
            txt = re.sub(r"TUNE = [0-9]+", f"TUNE = {200}", txt)
            txt = re.sub(r"DRAWS = [0-9]+", f"DRAWS = {200}", txt)
            txt = re.sub(r"CHAINS = [0-9]+", f"CHAINS = {1}", txt)

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
    success = True
    with open(notebook_filename) as f:
        notebook = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=-1)

    print(f"running: {notebook_filename}")
    try:
        ep.preprocess(
            notebook, {"metadata": {"path": f"notebooks/{version}"}}
        )
    except CellExecutionError as e:
        msg = f"error while running: {notebook_filename}\n\n"
        msg += e.traceback
        msg += f"removing: {notebook_filename}\n\n"
        print(msg)
        # os.remove(notebook_filename)
        success = False
    else:
        with open(notebook_filename, mode="wt") as f:
            nbformat.write(notebook, f)
        subprocess.check_call(f"git add {notebook_filename} -f", shell=True)

    return success


def get_toi_from_cli():
    """Get the TOI number from the CLI and return it"""
    if len(sys.argv) < 2:
        raise RuntimeError("you must give a TOI number")
    toi_number = int(sys.argv[1])
    return toi_number


def main():
    toi_number = get_toi_from_cli()
    notebook_filename = create_toi_notebook_from_template_notebook(toi_number)
    execute_toi_notebook(notebook_filename)


if __name__ == '__main__':
    main()
