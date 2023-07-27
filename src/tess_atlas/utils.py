import multiprocessing as mp
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Tuple, Union

import matplotlib.pyplot as plt
import tabulate

from .file_management import mkdir

CWD = os.getcwd()


def set_ipython_backend():
    from IPython import get_ipython

    ipy = get_ipython()
    if ipy is not None:
        ipy.magic('config InlineBackend.figure_format = "retina"')
        ipy.magic("matplotlib inline")
        ipy.magic("load_ext autoreload")
        ipy.magic("autoreload 2")


def notebook_initalisations(default=CWD):
    """Initialise the notebook environment."""
    set_global_environ_vars(default)

    try:
        mp.set_start_method("fork")
    except RuntimeError:  # "Multiprocessing context already set"
        pass

    # Warning
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    set_ipython_backend()
    set_plotting_style()
    print_global_environ_vars()


def print_global_environ_vars():
    """Print the global environment variables."""
    from IPython import display, get_ipython

    data = [
        ["Python version", sys.version],
        ["Current working directory", Path.cwd()],
        ["JOBFS", os.environ.get("JOBFS", "None")],
        ["THEANO_FLAGS", os.environ.get("THEANO_FLAGS")],
        ["OMP_NUM_THREADS", os.environ.get("OMP_NUM_THREADS")],
        ["IPYTHONDIR", os.environ.get("IPYTHONDIR")],
    ]

    if get_ipython() is not None:
        table = tabulate.tabulate(data, tablefmt="html")
        display.display(display.HTML(table))
    else:
        print(tabulate.tabulate(data, tablefmt="simple"))


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


def set_global_environ_vars(default=CWD):
    # Don't use the schmantzy progress bar
    os.environ["EXOPLANET_NO_AUTO_PBAR"] = "true"
    # Turn off ploomber stats
    os.environ["PLOOMBER_STATS_ENABLED"] = "false"
    # Use scratch space for theano cache
    theano_base, theano_comp = get_theano_cache(default)
    os.environ[
        "THEANO_FLAGS"
    ] = f"compiledir={theano_comp},base_compiledir={theano_base}"
    # Set the dir for IPython
    os.environ["IPYTHONDIR"] = mkdir(
        os.path.join(get_cache_dir(default), "ipython")
    )


def get_theano_cache(default=CWD) -> Tuple[str, str]:
    # make sure that THEANO has cache dir for each thread (prevent locking issues)
    base_dir = mkdir(os.path.join(get_cache_dir(default), "theano_basedir"))
    compile_dir = mkdir(
        os.path.join(get_cache_dir(default), "theano_compiledir")
    )
    return base_dir, compile_dir


def get_cache_dir(default=CWD) -> str:
    # JOBFS: ozstar specific scratch space
    cache_base = os.environ.get("JOBFS", default=default)
    cache = os.path.join(cache_base, ".tess-atlas-cache", str(os.getpid()))
    return mkdir(cache)


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
