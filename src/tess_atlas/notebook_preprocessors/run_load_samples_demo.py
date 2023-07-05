# -*- coding: utf-8 -*-
"""Execute the load_samples_demo.ipynb and save it in the outdir"""

# TODO: test this with notebooks
import logging
import os

import jupytext
import pkg_resources

from tess_atlas.utils import RUNNER_LOGGER_NAME

from .notebook_executor import execute_ipynb
from .paths import LOAD_SAMPLES_DEMO_FNAME

runner_logger = logging.getLogger(RUNNER_LOGGER_NAME)


def get_load_samples_demo_notebook_filename():
    """Write demo notebook in notebooks/load_samples_demo.ipynb"""
    ipynb_filename = os.path.basename(LOAD_SAMPLES_DEMO_FNAME).replace(
        ".py", ".ipynb"
    )
    ipynb_filename = os.path.join(f"./notebooks", ipynb_filename)
    py_pointer = jupytext.read(LOAD_SAMPLES_DEMO_FNAME, fmt="py:light")
    jupytext.write(py_pointer, ipynb_filename)
    return ipynb_filename


def main():
    ipynb_fn = get_load_samples_demo_notebook_filename()
    successful_operation = execute_ipynb(ipynb_fn)
    if successful_operation:
        runner_logger.info(f"Preprocessed {ipynb_fn}")
    else:
        runner_logger.warning(f"Couldnt process {ipynb_fn}")


if __name__ == "__main__":
    main()
