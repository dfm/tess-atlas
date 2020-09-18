#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Module to create and execute the notebooks for all existing TOIs

Uses `runt_toi.py` to create and execute the notebooks for the TOIs obtained from
https://exofop.ipac.caltech.edu/tess/

"""
from __future__ import division, print_function

import os
from multiprocessing import Pool
from subprocess import check_call

import numpy as np
import pandas as pd

TOI_DATABASE = (
    "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
)


def run_toi(toi_id):
    print(f"running {toi_id}")
    os.environ["THEANO_FLAGS"] = f"compiledir=./cache/{os.getpid()}"
    check_call(f"python tess_atlas/run_toi.py {toi_id}", shell=True)


def main():
    tois = pd.read_csv(TOI_DATABASE)
    toi_ids = np.floor(
        np.array(tois.groupby("TIC ID").first().sort_values("TOI").TOI)
    ).astype(int)
    np.random.shuffle(toi_ids)
    with Pool(8) as pool:
        pool.map(run_toi, toi_ids)


if __name__ == "__main__":
    main()
