import os
import re
from typing import Optional

import jupytext
import pkg_resources

from ..data.tic_entry import TICEntry
from ..tess_atlas_version import __version__

TOI_TEMPLATE_FNAME = "notebook_templates/toi_template.py"


def get_template_contents() -> str:
    """Gets str contents of toi_template.ipynb"""
    # get path to toi_template.py
    template_py_filename = pkg_resources.resource_filename(
        "tess_atlas", TOI_TEMPLATE_FNAME
    )
    # Converts toi_template.py --> toi_template.ipynb
    template_ipynb_filename = template_py_filename.replace(".py", ".ipynb")
    template_py_pointer = jupytext.read(template_py_filename, fmt="py:light")
    jupytext.write(template_py_pointer, template_ipynb_filename)

    # reads toi_template.ipynb contents
    with open(template_ipynb_filename, "r") as f:
        txt = f.read()
    return txt


def create_toi_notebook(
    toi_number: int, notebook_filename: str, quickrun: bool
):
    with open(notebook_filename, "w") as f:
        txt = get_template_contents()
        txt = txt.replace("{{{TOINUMBER}}}", f"{toi_number}")
        txt = txt.replace("{{{VERSIONNUMBER}}}", f"'{__version__}'")
        if quickrun:
            txt = re.sub(r"tune=[0-9]+", f"tune={5}", txt)
            txt = re.sub(r"draws=[0-9]+", f"draws={10}", txt)
            txt = re.sub(r"chains=[0-9]+", f"chains={1}", txt)
            txt = re.sub(r"cores=[0-9]+", f"cores={1}", txt)
        f.write(txt)


def download_toi_data(toi_number: int, notebook_dir: str):
    curr_dir = os.getcwd()
    os.chdir(notebook_dir)
    tic_data = TICEntry.load(toi_number)
    tic_data.save_data()
    os.chdir(curr_dir)


def create_toi_notebook_from_template_notebook(
    toi_number: int,
    outdir: Optional[str] = "notebooks",
    quickrun: Optional[bool] = False,
    setup: Optional[bool] = False,
):
    """Creates a jupyter notebook for the TOI

    Args:
        toi_number: int
            The TOI Id number
        quickrun: bool
            If True changes sampler settings to run the notebooks faster
            (useful for testing/debugging -- produces non-scientific results)
        outdir: str
            Base outdir for TOI. Notebook will be saved at
            {outdir}/{tess_atlas_version}/toi_{toi_number}.ipynb}
        setup: bool
            If true also downloads data needed for analysis and caches data.

    Returns:
        notebook_filename: str
            The filepath of the generated notebook
    """
    notebook_filename = os.path.join(
        outdir, f"{__version__}/toi_{toi_number}.ipynb"
    )
    notebook_dir = os.path.dirname(notebook_filename)
    os.makedirs(notebook_dir, exist_ok=True)

    create_toi_notebook(toi_number, notebook_filename, quickrun)

    if setup:
        download_toi_data(toi_number, notebook_dir)

    return notebook_filename
