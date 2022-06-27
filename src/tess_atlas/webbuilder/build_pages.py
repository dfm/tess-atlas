"""
Copy docs templates to outdir
Make summary plots + page
Run loader example
Build jupyter-book
"""
import os
import subprocess
import shutil
from typing import Optional
from ..file_management import copy_tree, make_tarfile


from tess_atlas.webbuilder.make_tois_homepage import make_menu_page


DIR = os.path.abspath(os.path.dirname(__file__))
TEMPLATES_DIR = f"{DIR}/template/"
NOTEBOOKS_DIR = "content/toi_notebooks"
MENU_PAGE = "content/toi_fits.rst"


def log(t, red=True):
    if red:
        t = f"\033[31m {t} \033[0m"
    print(t)


class PageBuilder:
    def __init__(self, notebook_src, builddir, rebuild: Optional[str] = None):
        """
        notebook_src/

        web1/
        ├── _build
        │   ├── _sources
        │   │   ├── content
        │   │   │   └── toi_notebooks
        │   └── content
        │       └── toi_notebooks
        │
        └── content
            └── toi_notebooks  (notebook_src copied here for building)

        """
        self.rebuild = rebuild
        self.notebook_src = notebook_src
        self.builddir = builddir
        self.building_notebook_dir = os.path.join(self.builddir, NOTEBOOKS_DIR)
        self.webdir = os.path.join(self.builddir, "_build")
        self.downloading_notebook = os.path.join(self.webdir, NOTEBOOKS_DIR)

    def setup(self):
        log(f"Website being built at {outdir}")
        copy_tree(TEMPLATES_DIR, outdir)  # templates --> outdir
        if rebuild != "soft":  # copy notebooks to new outdir
            copy_tree(notebooks_dir, new_notebook_dir)

    def tar_website(self):
        make_tarfile("tess_atlas_pages.tar.gz", source_dir=self.webdir)

    def build():
        pass


def sphinx_build_pages(outdir, webdir):
    command = f"sphinx-build -b html -j auto {outdir} {webdir}"
    log(f"Running >>>", red=False)
    log(command)
    subprocess.run(command, shell=True, check=True)


def build_webdir_structure(
    outdir, notebooks_dir, new_notebook_dir, rebuild=False
):
    log(f"Website being built at {outdir}")
    if rebuild:
        shutil.rmtree(outdir)
    os.makedirs(outdir, exist_ok=True)
    copy_tree(TEMPLATES_DIR, outdir)  # copy templates to outdir
    src = os.path.abspath(notebooks_dir)
    link = os.path.abspath(new_notebook_dir)
    os.symlink(src, link)


def make_book(
    outdir: str, notebooks_dir: str, rebuild: str, update_api_files=False
):
    # make + check for dirs
    outdir_present = os.path.isdir(outdir)
    new_notebook_dir = os.path.join(outdir, NOTEBOOKS_DIR)
    webdir = os.path.join(outdir, "_build")
    notebook_downloaddir = os.path.join(webdir, "_sources", NOTEBOOKS_DIR)

    if rebuild or outdir_present is False:
        build_webdir_structure(
            outdir, notebooks_dir, new_notebook_dir, rebuild
        )
    else:
        log(f"Website being updated at {outdir}")

    # make homepage
    make_menu_page(
        notebook_regex=os.path.join(new_notebook_dir, "toi_*.ipynb"),
        path_to_menu_page=os.path.join(outdir, MENU_PAGE),
    )

    # build book
    sphinx_build_pages(outdir, webdir)
    if update_api_files:
        copy_tree(notebooks_dir, notebook_downloaddir)
        print("\nAPI files copied")

    # tar pages
    make_tarfile("tess_atlas_pages.tar.gz", source_dir=webdir)

    log("Done! ")
