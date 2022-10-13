import os
import re
from typing import Optional

import jupytext
import nbformat
import pkg_resources

from ..data.tic_entry import TICEntry
from ..tess_atlas_version import __version__
from .paths import TOI_TEMPLATE_FNAME, TRANSIT_MODEL


def convert_py_to_ipynb(pyfile):
    ipynb = pyfile.replace(".py", ".ipynb")
    template_py_pointer = jupytext.read(pyfile, fmt="py:light")
    jupytext.write(template_py_pointer, ipynb)


def get_file_contents(path) -> str:
    with open(path, "r") as f:
        txt = f.read()
    return txt


def create_toi_notebook(
    toi_number: int, notebook_filename: str, quickrun: bool
):
    pyfile = notebook_filename.replace(".ipynb", ".py")
    with open(pyfile, "w") as f:
        txt = get_file_contents(TOI_TEMPLATE_FNAME)
        txt = txt.replace("{{{TOINUMBER}}}", f"{toi_number}")
        txt = txt.replace("{{{VERSIONNUMBER}}}", f"'{__version__}'")
        txt = txt.replace(
            "{{{TRANSIT_MODEL_CODE}}}", get_file_contents(TRANSIT_MODEL)
        )
        if quickrun:
            txt = re.sub(r"tune=[0-9]+", f"tune={5}", txt)
            txt = re.sub(r"draws=[0-9]+", f"draws={10}", txt)
            txt = re.sub(r"chains=[0-9]+", f"chains={1}", txt)
            txt = re.sub(r"cores=[0-9]+", f"cores={1}", txt)
            txt = txt.replace(
                "init_params(planet_transit_model, **params)",
                "init_params(planet_transit_model, **params, quick=True)",
            )
        f.write(txt)

    # convert to ipynb
    convert_py_to_ipynb(pyfile)
    os.remove(pyfile)

    # ensure notebook is valid
    notebook = nbformat.read(notebook_filename, as_version=4)
    nbformat.validate(notebook)


def safe_create_toi_notebook(
    toi_number: int,
    notebook_filename: str,
    quickrun: bool,
    attempts: Optional[int] = 5,
):
    for i in range(attempts):
        while True:
            try:
                create_toi_notebook(toi_number, notebook_filename, quickrun)
                break
            except Exception:
                continue


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

    safe_create_toi_notebook(toi_number, notebook_filename, quickrun)

    if setup:
        download_toi_data(toi_number, notebook_dir)

    return notebook_filename
