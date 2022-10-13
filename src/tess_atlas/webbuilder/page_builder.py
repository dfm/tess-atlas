"""
Copy docs templates to outdir
Make summary plots + page
Run loader example
Build jupyter-book
"""
import os
import shutil
import subprocess
from typing import Optional

from tess_atlas.utils import setup_logger
from tess_atlas.webbuilder.make_run_stats_page import make_stats_page
from tess_atlas.webbuilder.make_tois_homepage import make_menu_page

from ..file_management import copy_tree, make_tarfile

logger = setup_logger("page builder")

DIR = os.path.abspath(os.path.dirname(__file__))
TEMPLATES_DIR = f"{DIR}/template/"
NOTEBOOKS_DIR = "content/toi_notebooks"
MENU_PAGE = "content/toi_fits.rst"
STATS_PAGE = "content/stats.html"


def log(t, red=True):
    if red:
        t = f"\033[31m {t} \033[0m"
    logger.info(t)


class PageBuilder:
    def __init__(
        self,
        notebook_src,
        builddir,
        rebuild: Optional[bool] = None,
        update_api_files: Optional[bool] = False,
    ):
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
        self.update_api_files = update_api_files

    def setup_build_dir(self):
        if self.rebuild and os.path.exists(self.builddir):
            shutil.rmtree(self.builddir)

        if os.path.exists(self.builddir):
            log(f"Website being updated at {self.builddir}")
        else:
            log(f"Website being built at {self.builddir}")
            os.makedirs(self.builddir, exist_ok=True)
            copy_tree(TEMPLATES_DIR, self.builddir)
            src = os.path.abspath(self.notebook_src)
            link = os.path.abspath(self.building_notebook_dir)
            os.symlink(src, link)

    def tar_website(self):
        make_tarfile("tess_atlas_pages.tar.gz", source_dir=self.webdir)

    def sphinx_build_pages(self):
        command = f"sphinx-build -b html -j auto {self.builddir} {self.webdir}"
        log(f"Running >>>", red=False)
        log(command)
        subprocess.run(command, shell=True, check=True)

    def build(self):
        # make homepage
        toi_regex = os.path.join(self.building_notebook_dir, "toi_*.ipynb")
        make_menu_page(
            notebook_regex=toi_regex,
            path_to_menu_page=os.path.join(self.builddir, MENU_PAGE),
        )
        try:
            make_stats_page(
                notebook_root=toi_regex,
                path_to_stats_page=os.path.join(self.builddir, STATS_PAGE),
            )
        except Exception as e:
            logger.error(f"Failed to make stats page: {e}")

        # build book
        self.sphinx_build_pages()

        if self.update_api_files:
            log("\nCopying API files (this will take some time)...")
            copy_tree(self.notebook_src, self.building_notebook_dir)

        # tar pages
        log("TARing webdir contents\n")
        make_tarfile("tess_atlas_pages.tar.gz", source_dir=self.webdir)

        log("Done!")


def make_book(builddir, notebook_dir, rebuild, update_api_files):
    p = PageBuilder(
        notebook_src=notebook_dir,
        builddir=builddir,
        rebuild=rebuild,
        update_api_files=update_api_files,
    )
    p.setup_build_dir()
    p.build()
