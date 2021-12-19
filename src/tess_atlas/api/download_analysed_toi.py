import subprocess
import logging
import argparse

logger = logging.getLogger("TESS-Atlas")

COMMAND = "wget -np -r {url}"
ROOT = "http://catalog.tess-atlas.cloud.edu.au/content/toi_notebooks"
NOTEBOOK = f"{ROOT}/toi_{{TOI}}.ipynb"
FILES = f"{ROOT}/toi_{{TOI}}_files/"

ERROR = (
    "TOI{toi} has not been analysed. "
    "If you require its analysis, please raise an issue "
    "https://github.com/dfm/tess-atlas/issues/new?title=TOI{toi}."
)


def get_urls(toi: int):
    notebook_url = NOTEBOOK.format(TOI=toi)
    files_url = FILES.format(TOI=toi)
    return [notebook_url, files_url]


def get_path(toi: int):
    urls = get_urls(toi)
    paths = [url.split("http://")[1] for url in urls]
    return paths[0]


def download(toi: int):
    urls = get_urls(toi)
    try:
        for url in urls:
            subprocess.run(
                COMMAND.format(url=url),
                shell=True,
                check=True,
                capture_output=True,
            )
        logger.info(f"Notebook and data saved: {get_path(toi)}")
    except subprocess.CalledProcessError:
        logger.error(ERROR.format(toi=toi))


def get_cli_args():
    """Get the TOI number from the CLI and return it"""
    parser = argparse.ArgumentParser(prog="download_toi")
    parser.add_argument(
        "toi_number",
        type=int,
        help="The TOI number to download data for (e.g. 103)",
    )
    args = parser.parse_args()
    return args.toi_number


def main():
    toi = get_cli_args()
    download(toi)
