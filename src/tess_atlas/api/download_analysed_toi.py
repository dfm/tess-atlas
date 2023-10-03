import argparse
import logging
import subprocess

from .. import __website__
from ..logger import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)

COMMAND = "wget -np -r {url}"
ROOT = f"{__website__}/toi_data"
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


def download_toi(toi: int):
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
