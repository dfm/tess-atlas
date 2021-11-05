# -*- coding: utf-8 -*-
"""Module one liner

This module does what....

Example usage:

"""
import logging
import os

import jupytext
import pkg_resources

from tess_atlas.tess_atlas_version import __version__
from tess_atlas.utils import RUNNER_LOGGER_NAME

from .notebook_executor import execute_ipynb

runner_logger = logging.getLogger(RUNNER_LOGGER_NAME)


def get_load_samples_demo_notebook_filename(version=None):
    """Write demo notebook in notebooks/{version}/load_samples_demo.ipynb"""
    py_filename = pkg_resources.resource_filename(
        __name__, "load_samples_demo.py"
    )
    ipynb_filename = os.path.basename(py_filename).replace(".py", ".ipynb")
    ipynb_filename = os.path.join(f"./notebooks/{version}", ipynb_filename)
    py_pointer = jupytext.read(py_filename, fmt="py:light")
    jupytext.write(py_pointer, ipynb_filename)
    return ipynb_filename


def main():
    version = __version__
    ipynb_fn = get_load_samples_demo_notebook_filename(version)
    successful_operation = execute_ipynb(ipynb_fn)
    if successful_operation:
        runner_logger.info(f"Preprocessed {ipynb_fn}")
    else:
        runner_logger.warning(f"Couldnt process {ipynb_fn}")


if __name__ == "__main__":
    main()
