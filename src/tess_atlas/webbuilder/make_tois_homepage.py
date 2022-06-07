#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Module to build home page for TOIs"""
import glob
import os

from jinja2 import Template


TOI_LINK = Template("`TOI {{toi_int}}  <toi_notebooks/{{toi_fname}}>`_")

IMAGE = Template(
    """.. figure:: toi_notebooks/{{rel_path}}
            :target: toi_notebooks/{{toi_fname}}

"""
)


def render_page_template(fname, page_data):
    with open(fname) as file_:
        template = Template(file_.read())
    return template.render(**page_data)


def get_toi_str_from_path(path):
    return get_toi_fname(path).split("_")[1]


def get_toi_fname(path):
    return os.path.basename(path).split(".")[0]


def get_toi_number(path):
    return int(get_toi_str_from_path(path))


def render_toi_data(path):
    fname = get_toi_fname(path)
    toi_int = get_toi_number(path)
    return TOI_LINK.render(toi_int=toi_int, toi_fname=fname)


def sort_files(files):
    return sorted(files, key=lambda x: get_toi_number(x))


def get_phase_plots(notebook_path, notebook_dir):
    toi_str = get_toi_str_from_path(notebook_path)
    phase_regex = os.path.join(notebook_dir, f"toi_{toi_str}_files/phase*.png")
    phase_plots = glob.glob(phase_regex)
    return [p.split(notebook_dir)[1] for p in phase_plots]


def render_image_data(notebook_path, notebook_dir):
    image_paths = get_phase_plots(notebook_path, notebook_dir)
    toi_fname = get_toi_fname(notebook_path)
    return [IMAGE.render(rel_path=p, toi_fname=toi_fname) for p in image_paths]


def split_notebooks(notebook_files, notebook_dir):
    with_plots, without_plots = [], []
    for notebook_path in notebook_files:
        if len(get_phase_plots(notebook_path, notebook_dir)) > 0:
            with_plots.append(notebook_path)
        else:
            without_plots.append(notebook_path)
    return with_plots, without_plots


def generate_page_data(notebook_regex):
    notebook_files = sort_files(glob.glob(notebook_regex))
    notebook_dir = os.path.dirname(notebook_regex)

    success_notebooks, failed_notebooks = split_notebooks(
        notebook_files, notebook_dir
    )
    num_fail, num_pass = len(failed_notebooks), len(success_notebooks)

    successful_data = {"TOI": ["Phase Plot"]}
    for notebook_path in success_notebooks:
        toi_data = render_toi_data(notebook_path)
        image_data = render_image_data(notebook_path, notebook_dir)
        successful_data[toi_data] = image_data

    failed_data = [render_toi_data(p) for p in failed_notebooks]

    return dict(
        total_number_tois=num_fail + num_pass,
        number_successful_tois=num_pass,
        number_failed_tois=num_fail,
        successful_tois=successful_data,
        failed_tois=failed_data,
    )


def make_menu_page(notebook_regex, path_to_menu_page):
    page_data = generate_page_data(notebook_regex)
    page_contents = render_page_template(path_to_menu_page, page_data)

    with open(path_to_menu_page, "w") as f:
        f.write(page_contents)
