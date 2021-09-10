#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Module to create and execute the notebooks for all existing TOIs

Uses `runt_toi.py` to create and execute the notebooks for the TOIs obtained from
https://exofop.ipac.caltech.edu/tess/

"""
from __future__ import division, print_function

import argparse
import logging
import os
from multiprocessing import Pool
from subprocess import check_call

import numpy as np
import pandas as pd

logging.getLogger().setLevel(logging.INFO)
TOI_DATABASE = (
    "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
)


def run_toi(args):
    # make sure that THEANO has cache dir for each thread (prevent locking issues)
    os.environ["THEANO_FLAGS"] = f"compiledir=./cache/{os.getpid()}"
    command = f"run_toi {args}"
    logging.info(f"Running {command}")
    check_call(command, shell=True)


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
    args = parser.parse_args()
    return args.outdir, args.toi_database


def main():
    outdir, toi_database = get_cli_args()
    tois = pd.read_csv(toi_database)
    toi_ids = np.floor(
        np.array(tois.groupby("TIC ID").first().sort_values("TOI").TOI)
    ).astype(int)
    logging.info(f"Number of TOIs to analyse: {len(toi_ids)}")
    np.random.shuffle(toi_ids)
    run_toi_args = [f"{toi_id} --outdir {outdir}" for toi_id in toi_ids]
    with Pool(8) as pool:
        pool.map(run_toi, run_toi_args)


if __name__ == "__main__":
    main()
