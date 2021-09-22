#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Module to build home page for TOIs"""
import glob
import os

from jinja2 import Template

HEADER = Template(
    """
---
## {doc}`TOI {{toi_int}} <toi_notebooks/{{toi_fname}}>`
"""
)

IMAGE = Template(
    """
```{image} toi_notebooks/{{rel_path}}
:name: TOI {{toi_int}}
```
"""
)

ERROR = """
```{error} Phase plot generation failed. There may be underlying errors.
```
"""


def get_toi_str_from_path(path):
    return get_toi_fname(path).split("_")[1]


def get_toi_fname(path):
    return os.path.basename(path).split(".")[0]


def get_toi_number(path):
    return int(get_toi_str_from_path(path))


def make_menu_page(notebook_regex, path_to_menu_page):
    notebook_files = glob.glob(notebook_regex)
    notebook_dir = os.path.dirname(notebook_regex)
    lines = []
    for notebook_path in sorted(notebook_files):
        fname = get_toi_fname(notebook_path)
        toi_int = get_toi_number(notebook_path)
        lines.append(HEADER.render(toi_int=toi_int, toi_fname=fname))

        toi_str = get_toi_str_from_path(notebook_path)
        phase_regex = os.path.join(
            notebook_dir, f"toi_{toi_str}_files/phase*.png"
        )
        phase_plots = glob.glob(phase_regex)

        if len(phase_plots) == 0:
            lines.append(ERROR)
        else:
            for phase_plot in phase_plots:
                rel_path = phase_plot.split(notebook_dir)[1]
                lines.append(IMAGE.render(rel_path=rel_path, toi_int=toi_int))

    with open(path_to_menu_page, "a") as f:
        f.writelines(lines)
