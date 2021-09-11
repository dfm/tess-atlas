import logging
import os

import nbformat
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor

from tess_atlas.utils import RUNNER_LOGGER_NAME


def execute_ipynb(notebook_filename: str):
    """
    :param notebook_filename: path of notebook to process
    :return: bool if notebook-preprocessing successful/unsuccessful
    """
    success = True
    with open(notebook_filename) as f:
        notebook = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=-1)
    runner_logger = logging.getLogger(RUNNER_LOGGER_NAME)
    runner_logger.info(f"Executing {notebook_filename}")
    try:
        # Note that path specifies in which folder to execute the notebook.
        run_path = os.path.dirname(notebook_filename)
        ep.preprocess(notebook, {"metadata": {"path": run_path}})
    except CellExecutionError as e:
        runner_logger.error(
            f"Preprocessing {notebook_filename} failed:\n\n {e.traceback}"
        )
        success = False
    finally:
        with open(notebook_filename, mode="wt") as f:
            nbformat.write(notebook, f)
    return success
