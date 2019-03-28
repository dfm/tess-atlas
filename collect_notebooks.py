#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob

version = sys.argv[1].strip()

lines = []
for filename in sorted(glob.glob("notebooks/{0}/*.ipynb".format(version))):
    toi = int(os.path.splitext(os.path.split(filename)[1])[0].split("-")[1])
    lines.append("<li><a href=\"toi-{0}.html\">TOI {0}</a></li>".format(toi))

with open("docs/notebooks/index.html.tpl", "r") as f:
    txt = f.read()

txt = txt.replace("{{{TOILIST}}}", "\n".join(lines))
with open("docs/notebooks/{0}/index.html".format(version), "w") as f:
    f.write(txt)
