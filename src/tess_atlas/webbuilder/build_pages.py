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
import tarfile

from tess_atlas.webbuilder.make_tois_homepage import make_menu_page

dist_log.set_verbosity(dist_log.INFO)
dist_log.set_threshold(dist_log.INFO)
DIR = os.path.abspath(os.path.dirname(__file__))
TEMPLATES_DIR = f"{DIR}/template/"
NOTEBOOKS_DIR = "content/toi_notebooks"
MENU_PAGE = "content/toi_fits.rst"


def make_tarfile(output_filename, source_dir):
    print(f"TARing {source_dir} -> {output_filename}")
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))


def sphinx_build_pages(outdir, webdir):
    command = f"sphinx-build -b html {outdir} {webdir}"
    print(f"Running >>> \n \033[31m {command} \033[0m")
    subprocess.run(command, shell=True, check=True)


def build_webdir_structure(
    outdir, notebooks_dir, new_notebook_dir, rebuild="hard"
):
    print(f"Website being built at {outdir}")
    os.makedirs(outdir, exist_ok=True)
    copy_tree(TEMPLATES_DIR, outdir)  # copy templates to outdir
    if rebuild != "soft":  # copy notebooks to new outdir
        copy_tree(notebooks_dir, new_notebook_dir)


def make_book(outdir: str, notebooks_dir: str, rebuild: str):
    outdir_present = os.path.isdir(outdir)
    new_notebook_dir = os.path.join(outdir, NOTEBOOKS_DIR)
    webdir = os.path.join(outdir, "_build")

    if rebuild or outdir_present is False:
        build_webdir_structure(
            outdir, notebooks_dir, new_notebook_dir, rebuild
        )
    else:
        print(f"Website being updated at {outdir}")

    make_menu_page(
        notebook_regex=os.path.join(new_notebook_dir, "toi_*.ipynb"),
        path_to_menu_page=os.path.join(outdir, MENU_PAGE),
    )

    sphinx_build_pages(outdir, webdir)

    make_tarfile("tess_atlas_pages.tar.gz", source_dir=webdir)

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
        type=str,
        default="",
        help="""'hard': Rebuild from scratch (even if some webpages exist).
        'soft': Rebuild pages w/o copying notebooks again.""",
    )
    args = parser.parse_args()
    return args.web_outdir, args.notebooks_dir, args.rebuild


def main():
    outdir, notebooks_dir, rebuild = get_cli_args()
    make_book(outdir=outdir, notebooks_dir=notebooks_dir, rebuild=rebuild)


if __name__ == "__main__":
    main()
