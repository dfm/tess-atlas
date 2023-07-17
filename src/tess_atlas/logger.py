import datetime
import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Union

LOGGER_NAME = "TESS-ATLAS"


def __get_logger_fname(name):
    return name.replace("-", "_").lower() + ".log"


LOG_FNAME = __get_logger_fname(LOGGER_NAME)


def get_notebook_logger(outdir=""):
    # Logging setup
    for logger_name in [
        "theano.gof.compilelock",
        "filelock",
        "lazylinker_c.py",
        "theano.tensor.opt",
        "exoplanet",
        "matplotlib",
        "urllib3",
        "arviz",
        "astropy",
        "lightkurve",
        "corner",
    ]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.ERROR)

    notebook_logger = setup_logger(LOGGER_NAME, outdir)
    return notebook_logger


def setup_logger(
    logger_name: str,
    outdir: Union[Optional[str], Path] = "",
    level=logging.INFO,
):
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    if logger.hasHandlers():
        return logger  # already set up

    # add custom formatter to root logger
    handler = logging.StreamHandler()
    formatter = DeltaTimeFormatter(
        "\033[92m[%(delta)s - %(name)s]\033[0m %(message)s"
    )

    # console logging
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)

    if outdir != "":  # setup file logging
        os.makedirs(outdir, exist_ok=True)
        fname = os.path.join(outdir, __get_logger_fname(logger_name))
        fh = logging.FileHandler(fname)
        fh.setFormatter(formatter)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
    return logger


class DeltaTimeFormatter(logging.Formatter):
    def format(self, record):
        duration = datetime.datetime.utcfromtimestamp(
            record.relativeCreated / 1000
        )
        record.delta = duration.strftime("%H:%M:%S")
        return super().format(record)


@contextmanager
def all_logging_disabled(highest_level=logging.CRITICAL):
    """
    A context manager that will prevent any logging messages
    triggered during the body from being processed.
    :param highest_level: the maximum logging level in use.
      This would only need to be changed if a custom level greater than CRITICAL
      is defined.
    """
    # two kind-of hacks here:
    #    * can't get the highest logging level in effect => delegate to the user
    #    * can't get the current module-level override => use an undocumented
    #       (but non-private!) interface

    previous_level = logging.root.manager.disable

    logging.disable(highest_level)

    try:
        yield
    finally:
        logging.disable(previous_level)
