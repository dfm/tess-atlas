# -*- coding: utf-8 -*-

import logging

import nbformat
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor


def execute_ipynb(notebook_filename: str, version: str):
    success = True
    with open(notebook_filename) as f:
        notebook = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=-1)

    logging.info(f"Executing {notebook_filename}")
    try:
        # Note that path specifies in which folder to execute the notebook.
        ep.preprocess(notebook, {"metadata": {"path": f"notebooks/{version}"}})
    except CellExecutionError as e:
        logging.error(
            f"Preprocessing {notebook_filename} failed:\n\n {e.traceback}"
        )
        success = False
    finally:
        with open(notebook_filename, mode="wt") as f:
            nbformat.write(notebook, f)
    return success
