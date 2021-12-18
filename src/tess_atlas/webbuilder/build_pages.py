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

TEMPLATES_DIR = f"{os.path.dirname(__file__)}/../../../docs/"
NOTEBOOKS_DIR = "content/toi_notebooks"
MENU_PAGE = "content/toi_fits.md"


def build_webdir_structure(outdir, notebooks_dir, new_notebook_dir):
    print(f"Website being built at {outdir}")
    os.makedirs(outdir, exist_ok=True)
    copy_tree(TEMPLATES_DIR, outdir)  # copy templates to outdir
    copy_tree(notebooks_dir, new_notebook_dir)  # copy notebooks to new outdir


def make_book(outdir: str, notebooks_dir: str, rebuild: bool):
    outdir_present = os.path.isdir(outdir)
    new_notebook_dir = os.path.join(outdir, NOTEBOOKS_DIR)

    if rebuild or outdir_present is False:
        build_webdir_structure(outdir, notebooks_dir, new_notebook_dir)
    else:
        print(f"Website being updated at {outdir}")

    make_menu_page(
        notebook_regex=os.path.join(new_notebook_dir, "toi_*.ipynb"),
        path_to_menu_page=os.path.join(outdir, MENU_PAGE),
    )
    subprocess.run(f"jupyter-book build {outdir}", shell=True, check=True)

    print(f"TARing contents")
    subprocess.run(
        f"tar -czf {os.path.dirname(outdir)}.tar {outdir}",
        shell=True,
        check=True,
    )
    print("Done! ")


def get_cli_args():
    """Get the TOI number from the CLI and return it"""
    parser = argparse.ArgumentParser(prog="build webpages")
    parser.add_argument("web_outdir", type=str, help="The weboutdir")
    parser.add_argument(
        "notebooks_dir", type=str, help="Directory with analysed notebooks"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",  # False by default
        help="Rebuild from scratch (even if some webpages exist).",
    )
    args = parser.parse_args()
    return args.web_outdir, args.notebooks_dir, args.rebuild


def main():
    outdir, notebooks_dir, rebuild = get_cli_args()
    make_book(outdir=outdir, notebooks_dir=notebooks_dir, rebuild=rebuild)


if __name__ == "__main__":
    main()
