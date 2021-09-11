import logging
import os
import sys

RUNNER_LOGGER_NAME = "TESS-ATLAS-RUNNER"
NOTEBOOK_LOGGER_NAME = "TESS-ATLAS"

runner_logger = logging.getLogger(RUNNER_LOGGER_NAME)
notebook_logger = logging.getLogger(NOTEBOOK_LOGGER_NAME)


def setup_logger(logger_name, outdir=""):
    logging.getLogger().setLevel(logging.INFO)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(sh)
    if outdir != "":
        os.makedirs(outdir, exist_ok=True)
        filename = os.path.join(outdir, f"{logger_name}_runner.log")
        fh = logging.FileHandler(filename)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger
