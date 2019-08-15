#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import subprocess
import numpy as np
import pandas as pd
from astropy.io import ascii

# ASTERO
tab = ascii.read("chaplin_table4.txt")
dwarfs = tab[(tab["Radius"] < 2) & (~tab["Radius"].mask)]
ids1 = np.array(dwarfs["KIC"], dtype=int)

tab = ascii.read("chaplin_table5.txt")
dwarfs = tab[(tab["Radius"] < 2) & (~tab["Radius"].mask)]
ids2 = np.array(dwarfs["KIC"], dtype=int)

ids = np.unique(np.concatenate((ids1, ids2)))

# CKS
# df = pd.read_csv("cks_physical_merged.csv")
# ids = np.unique(np.array(df.id_kic, dtype=int))

np.random.shuffle(ids)
splits = np.array_split(ids, 10)

procs = []
files = []
for i, split in enumerate(splits):
    cmd = "python run_fit.py {0}".format(" ".join(map("{0}".format, split)))
    print(cmd)

    logfp = open(os.path.join("astero_results", "{0:03d}.stdout.log".format(i)), "w")
    errfp = open(os.path.join("astero_results", "{0:03d}.stderr.log".format(i)), "w")
    procs.append(subprocess.Popen(cmd, shell=True, stdout=logfp, stderr=errfp))
    files += [logfp, errfp]


try:
    while not all(p is None for p in procs):

        time.sleep(10)

        for i, proc in enumerate(procs):
            if proc is None:
                continue
            code = proc.poll()
            if code is not None:
                print(i, code)
                procs[i] = None

finally:

    for proc in procs:
        if proc is not None:
            proc.kill()
    for f in files:
        f.close()
