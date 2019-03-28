#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
import pandas as pd
from multiprocessing import Pool
from subprocess import check_call

def run_toi(toi_id):
    check_call("python run_toi.py {0}".format(toi_id), shell=True)

tois = pd.read_csv("https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv")
toi_ids = np.floor(np.array(tois.groupby("TIC ID").first().sort_values("TOI").TOI)).astype(int)
np.random.shuffle(toi_ids)

with Pool(8) as pool:
    pool.map(run_toi, toi_ids)
