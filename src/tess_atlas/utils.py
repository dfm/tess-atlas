# -*- coding: utf-8 -*-

import logging
import multiprocessing as mp
import os
import warnings
import sys

import matplotlib.pyplot as plt
import nbformat
from IPython import get_ipython
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor

from .logger import RUNNER_LOGGER_NAME, NOTEBOOK_LOGGER_NAME, setup_logger
from .tess_atlas_version import __version__


def execute_ipynb(notebook_filename: str):
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


def notebook_initalisations():
    get_ipython().magic('config InlineBackend.figure_format = "retina"')

    try:
        mp.set_start_method("fork")
    except RuntimeError:  # "Multiprocessing context already set"
        pass

    # Don't use the schmantzy progress bar
    os.environ["EXOPLANET_NO_AUTO_PBAR"] = "true"

    # Warning
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Logging setup
    for logger_name in [
        "theano.gof.compilelock",
        "exoplanet",
        "matplotlib",
        "urllib3",
        "arviz",
        "astropy",
        "lightkurve",
    ]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.ERROR)

    notebook_logger = setup_logger(NOTEBOOK_LOGGER_NAME)
    notebook_logger.setLevel(logging.INFO)

    plt.style.use("default")
    plt.rcParams["savefig.dpi"] = 100
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["font.size"] = 16
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Liberation Sans"]
    plt.rcParams["font.cursive"] = ["Liberation Sans"]
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["image.cmap"] = "inferno"
