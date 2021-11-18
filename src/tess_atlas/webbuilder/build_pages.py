"""
Copy docs templates to outdir
Make summary plots + page
Run loader example
Build jupyter-book
"""
import argparse
import os
import subprocess
from distutils import log as dist_log
from distutils.dir_util import copy_tree

from tess_atlas.webbuilder.make_tois_homepage import make_menu_page

dist_log.set_verbosity(dist_log.INFO)
dist_log.set_threshold(dist_log.INFO)

DEFAULT_TEMPLATES_DIR = f"{os.path.dirname(__file__)}/../../../docs/"
NOTEBOOKS_DIR = "content/toi_notebooks"
MENU_PAGE = "content/toi_fits.md"


def make_book(outdir, notebooks_dir, templates_dir=DEFAULT_TEMPLATES_DIR):
    outdir_present = os.path.isdir(outdir)
    new_notebook_dir = os.path.join(outdir, NOTEBOOKS_DIR)

    if outdir_present:
        print(f"Outdir ({os.path.abspath(outdir)}) exists")
        input_var = input("press [r] to rebuild, [u] to update [r/u]: ")
        if input_var == "r":
            print(f"Website being built at {outdir}")
            os.makedirs(outdir, exist_ok=True)
            copy_tree(templates_dir, outdir)  # copy templates to outdir
            copy_tree(
                notebooks_dir, new_notebook_dir
            )  # copy notebooks to new outdir
        else:
            print(f"Website being updated at {outdir}")

    make_menu_page(
        notebook_regex=os.path.join(new_notebook_dir, "toi_*.ipynb"),
        path_to_menu_page=os.path.join(outdir, MENU_PAGE),
    )
    subprocess.run(f"jupyter-book build {outdir}", shell=True, check=True)


def get_cli_args():
    """Get the TOI number from the CLI and return it"""
    parser = argparse.ArgumentParser(prog="build webpages")
    parser.add_argument("web_outdir", type=str, help="The weboutdir")
    parser.add_argument(
        "notebooks_dir", type=str, help="Directory with analysed notebooks"
    )
    parser.add_argument(
        "templates_dir",
        default=DEFAULT_TEMPLATES_DIR,
        type=str,
        help="Directory with templates",
    )
    args = parser.parse_args()
    return args.web_outdir, args.notebooks_dir, args.templates_dir


def main():
    outdir, notebooks_dir, templates_dir = get_cli_args()
    make_book(
        outdir=outdir, notebooks_dir=notebooks_dir, templates_dir=templates_dir
    )


if __name__ == "__main__":
    main()
