#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Module to create and execute the notebooks for all existing TOIs

Uses `run_toi.py` to create and execute the notebooks for the TOIs obtained from
https://exofop.ipac.caltech.edu/tess/

"""
from __future__ import division, print_function

import argparse
import logging
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from tess_atlas.notebook_preprocessors.run_toi import run_toi
from tess_atlas.tess_atlas_version import __version__
from tess_atlas.utils import RUNNER_LOGGER_NAME, setup_logger

TOI_DATABASE = (
    "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
)


def run_toi_in_pool(kwargs):
    logger = logging.getLogger(RUNNER_LOGGER_NAME)
    logger.info(f"Running TOI with {kwargs}")
    run_status, run_duration = run_toi(**kwargs)
    logger.info(
        f"TOI {kwargs['toi_number']} Passed: {run_status} ({run_duration})"
    )


def get_cli_args():
    """Get the ___ from the CLI and return it"""
    parser = argparse.ArgumentParser(prog="run_tois")
    default_outdir = os.path.join(os.getcwd(), "notebooks")
    parser.add_argument(
        "--outdir",
        default=default_outdir,
        type=str,
        help="The outdir to save notebooks (default: cwd/notebooks)",
    )
    parser.add_argument(
        "--toi_database",
        default=TOI_DATABASE,
        type=str,
        help="The csv with TOI numbers (has to have the columns [`TIC ID, TOI`])",
    )
    parser.add_argument(
        "--quickrun",
        action="store_true",  # False by default
        help="Run with reduced sampler settings (useful for debugging)",
    )
    parser.add_argument(
        "--setup",
        action="store_true",  # False by default
        help="Create notebooks and download data for analysis (dont execute notebooks)",
    )
    args = parser.parse_args()
    return args.outdir, args.toi_database, args.quickrun, args.setup


def main():
    outdir, toi_database, quickrun, setup = get_cli_args()
    logger = setup_logger(
        outdir=os.path.join(outdir, __version__),
        logger_name=RUNNER_LOGGER_NAME,
    )
    tois = pd.read_csv(toi_database)
    toi_ids = np.floor(
        np.array(tois.groupby("TIC ID").first().sort_values("TOI").TOI)
    ).astype(int)
    logger.info(f"Number of TOIs to analyse: {len(toi_ids)}")
    np.random.shuffle(toi_ids)
    run_toi_args = [
        dict(toi_number=toi_id, outdir=outdir, quickrun=quickrun, setup=setup)
        for toi_id in toi_ids
    ]
    with Pool(8) as pool:
        pool.map(run_toi_in_pool, run_toi_args)


if __name__ == "__main__":
    main()
