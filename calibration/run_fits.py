#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = []

import numpy as np
from astropy.io import ascii

tab = ascii.read("calibration/chaplin_table4.txt")
dwarfs = tab[(tab["Radius"] < 2) & (~tab["Radius"].mask)]
ids = np.array(dwarfs["KIC"], dtype=int)

for split in np.array_split(ids, 10):
    cmd = "python run_fit.py {0}".format(" ".join(map("{0}".format, split)))
    print(cmd)
