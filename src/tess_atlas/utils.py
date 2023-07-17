import datetime
import logging
import multiprocessing as mp
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
from IPython import get_ipython


def notebook_initalisations():
    ipy = get_ipython()
    if ipy is not None:
        ipy.magic('config InlineBackend.figure_format = "retina"')

    try:
        mp.set_start_method("fork")
    except RuntimeError:  # "Multiprocessing context already set"
        pass

    # Don't use the schmantzy progress bar
    os.environ["EXOPLANET_NO_AUTO_PBAR"] = "true"

    # Warning
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    set_plotting_style()
    set_theano_cache()


def set_plotting_style():
    plt.style.use("default")
    plt.rcParams["savefig.dpi"] = 100
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["font.size"] = 16
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Liberation Sans"]
    plt.rcParams["font.cursive"] = ["Liberation Sans"]
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["image.cmap"] = "inferno"


def set_theano_cache():
    # make sure that THEANO has cache dir for each thread (prevent locking issues)
    theano_cache = os.path.join(
        get_cache_dir(), "theano_cache", str(os.getpid())
    )
    os.makedirs(theano_cache, exist_ok=True)
    os.environ["THEANO_FLAGS"] = f"compiledir={theano_cache}"


def get_cache_dir(default="./"):
    # ozstar specific scratch space
    return os.environ.get("JOBFS", default=default)


def grep_toi_number(str) -> Union[int, None]:
    """Extract TOI number from string using regex
    "http://localhost:63342/tess-atlas/tests/out_webtest/html/_build/content/toi_notebooks/toi_101.html"
    "__website__/content/toi_notebooks/toi_101.html"
    "run_toi(101)"

    """
    regex = r"toi_(\d+)|run_toi\((\d+)\)"
    toi_number = re.search(regex, str)
    if toi_number is None:
        return None
    return int(toi_number.group(1))
