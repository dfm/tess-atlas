#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import re
import os
import sys
import nbformat
import subprocess
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError

__version__ = "0.1.0"

if len(sys.argv) < 2:
    raise RuntimeError("you must give a TOI number")

toi_number = int(sys.argv[1])
filename = "notebooks/{0}/toi-{1}.ipynb".format(__version__, toi_number)
os.makedirs(os.path.dirname(filename), exist_ok=True)

with open("template.ipynb", "r") as f:
    txt = f.read()
    txt = txt.replace("{{{TOINUMBER}}}", "{0}".format(toi_number))
    txt = txt.replace("{{{VERSIONNUMBER}}}", "{0}".format(__version__))
    txt = re.sub(r"toi_num = [0-9]+", "toi_num = {0}".format(toi_number),
                 txt)

with open(filename, "w") as f:
    f.write(txt)

with open(filename) as f:
    notebook = nbformat.read(f, as_version=4)

ep = ExecutePreprocessor(timeout=-1)

print("running: {0}".format(filename))
try:
    ep.preprocess(notebook, {"metadata":
                             {"path": "notebooks/{0}".format(__version__)}})
except CellExecutionError as e:
    msg = "error while running: {0}\n\n".format(filename)
    msg += e.traceback
    print(msg)
finally:
    with open(filename, mode="wt") as f:
        nbformat.write(notebook, f)

subprocess.check_call("git add {0}".format(filename), shell=True)
