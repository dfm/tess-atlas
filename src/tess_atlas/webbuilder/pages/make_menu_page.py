"""Module to build menu page for TOIs

The page will contain links to the TOI notebooks, and a thumbnail of the phase plot
See http://catalog.tess-atlas.cloud.edu.au/content/toi_fits.html


#TODO: This will hopefully be replaced with the AnalysisSummary class + itables
"""
import glob
import os

from tess_atlas.utils import grep_toi_number

from ...data.exofop import EXOFOP_DATA, constants
from .templates import IMAGE, TOI_LINK, render_page_template

CATEGORISED_TOIS = EXOFOP_DATA.get_categorised_toi_lists()
CATEGRIES = [constants.NORMAL, constants.MULTIPLANET, constants.SINGLE_TRANSIT]


def get_toi_str_from_path(path):
    return get_toi_fname(path).split("_")[1]


def get_toi_fname(path):
    return os.path.basename(path).split(".")[0]


def render_toi_data(path):
    fname = get_toi_fname(path)
    toi_int = grep_toi_number(path)
    return TOI_LINK.render(toi_int=toi_int, toi_fname=fname)


def sort_files(files):
    return sorted(files, key=lambda x: grep_toi_number(x))


def get_phase_plots(notebook_path, notebook_dir):
    toi_str = get_toi_str_from_path(notebook_path)
    phase_regex = os.path.join(
        notebook_dir, f"toi_{toi_str}_files/phase*_lowres.png"
    )
    phase_plots = glob.glob(phase_regex)
    if len(phase_plots) > 0:
        return [p.split(notebook_dir)[1] for p in phase_plots]
    return []


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


def generate_number_data(successful_data, failed_data):
    numbers = {
        constants.SINGLE_TRANSIT: EXOFOP_DATA.get_counts().single_transit,
        constants.MULTIPLANET: EXOFOP_DATA.get_counts().multiplanet,
        constants.NORMAL: EXOFOP_DATA.get_counts().normal,
    }
    numbers["total"] = EXOFOP_DATA.get_counts().all
    total_done, total_fail = 0, 0
    for type in CATEGRIES:
        numbers[f"{type}_done"] = len(successful_data[type].keys()) - 1
        numbers[f"{type}_fail"] = len(failed_data[type])
        total_done += numbers[f"{type}_done"]
        total_fail += numbers[f"{type}_fail"]
    numbers.update(dict(done=total_done, fail=total_fail))
    return numbers


def get_toi_category(notebook_path):
    toi_number = grep_toi_number(notebook_path)
    if toi_number in CATEGORISED_TOIS.normal:
        return constants.NORMAL
    elif toi_number in CATEGORISED_TOIS.multiplanet:
        return constants.MULTIPLANET
    elif toi_number in CATEGORISED_TOIS.single_transit:
        return constants.SINGLE_TRANSIT
    return "NA"


def generate_page_data(notebook_regex):
    """
    required data:
    - "number" dict with keys {
        done, fail, single, multi, norm,
        fail_single, fail_multi, fail_norm,
        done_single, done_norm, done_multi
    }
    - "successful_tois" dict of dict {
        "normal" {toi_link: toi_phase_plot},
        "single" {toi_link: toi_phase_plot},
        "multi" {toi_link: toi_phase_plot},
    }
    - "failed_tois" dict of {
        "normal" [toi_link],
        "single" [toi_link]
        "multi" [toi_link]
    }
    """
    notebook_files = sort_files(glob.glob(notebook_regex))
    notebook_dir = os.path.dirname(notebook_regex)

    success_notebooks, failed_notebooks = split_notebooks(
        notebook_files, notebook_dir
    )
    num_fail, num_pass = len(failed_notebooks), len(success_notebooks)

    successful_data = {k: {"TOI": ["Phase Plot"]} for k in CATEGRIES}
    failed_data = {k: [] for k in CATEGRIES}
    failed_data["NA"] = []

    for notebook_path in success_notebooks:
        toi_data = render_toi_data(notebook_path)
        image_data = render_image_data(notebook_path, notebook_dir)
        toi_type = get_toi_category(notebook_path)
        successful_data[toi_type][toi_data] = image_data

    for notebook_path in failed_notebooks:
        toi_type = get_toi_category(notebook_path)
        failed_data[toi_type].append(render_toi_data(notebook_path))

    number = generate_number_data(successful_data, failed_data)
    number["fail"], number["done"] = num_fail, num_pass
    return dict(
        number=number,
        successful_tois=successful_data,
        failed_tois=failed_data,
    )


def make_menu_page(notebook_regex, path_to_menu_page):
    page_data = generate_page_data(notebook_regex)
    page_contents = render_page_template(path_to_menu_page, page_data)

    with open(path_to_menu_page, "w") as f:
        f.write(page_contents)
