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

ERROR_HEADER = """
---
# Erroneous fits:
"""

ERROR = Template(
    """
- [TOI {{toi_int}}](toi_notebooks/{{toi_fname}})
"""
)


def get_toi_str_from_path(path):
    return get_toi_fname(path).split("_")[1]


def get_toi_fname(path):
    return os.path.basename(path).split(".")[0]


def get_toi_number(path):
    return int(get_toi_str_from_path(path))


def sort_files(files):
    return sorted(files, key=lambda x: get_toi_number(x))


def get_phase_plots(notebook_path, notebook_dir):
    toi_str = get_toi_str_from_path(notebook_path)
    phase_regex = os.path.join(notebook_dir, f"toi_{toi_str}_files/phase*.png")
    phase_plots = glob.glob(phase_regex)
    return phase_plots


def split_notebooks(notebook_files, notebook_dir):
    with_plots, without_plots = [], []
    for notebook_path in notebook_files:
        if len(get_phase_plots(notebook_path, notebook_dir)) > 0:
            with_plots.append(notebook_path)
        else:
            without_plots.append(notebook_path)
    return with_plots, without_plots


def make_menu_page(notebook_regex, path_to_menu_page):
    notebook_files = sort_files(glob.glob(notebook_regex))
    notebook_dir = os.path.dirname(notebook_regex)

    success_notebooks, failed_notebooks = split_notebooks(
        notebook_files, notebook_dir
    )

    lines = []

    for notebook_path in success_notebooks:
        fname = get_toi_fname(notebook_path)
        toi_int = get_toi_number(notebook_path)
        phase_plots = get_phase_plots(notebook_path, notebook_dir)
        lines.append(HEADER.render(toi_int=toi_int, toi_fname=fname))
        for phase_plot in phase_plots:
            rel_path = phase_plot.split(notebook_dir)[1]
            lines.append(IMAGE.render(rel_path=rel_path, toi_int=toi_int))

    lines.append(ERROR_HEADER)
    for notebook_path in failed_notebooks:
        fname = get_toi_fname(notebook_path)
        toi_int = get_toi_number(notebook_path)
        lines.append(ERROR.render(toi_int=toi_int, toi_fname=fname))

    with open(path_to_menu_page, "a") as f:
        f.writelines(lines)
