"""Module to parse the metadata from the TOI notebooks"""
import json
import logging
import os
import re
from typing import Dict, Union

import numpy as np
import pandas as pd

from .... import __website__
from ....data.exofop import EXOFOP_DATA
from ....data.exofop.constants import MULTIPLANET, NORMAL, SINGLE_TRANSIT
from ....data.planet_candidate import CLASSIFICATION_SHORTHAND
from ....file_management import (
    INFERENCE_DATA_FNAME,
    LC_DATA_FNAME,
    PROFILING_CSV,
    TIC_CSV,
    TOI_DIR,
    read_last_n_lines,
)
from ....logger import LOG_FNAME
from ....plotting.labels import THUMBNAIL_PLOT
from ....utils import grep_toi_number
from .analysis_status import Status

URL_BASE = f"{__website__}/content/toi_notebooks"
NOTEBOOK_URL = URL_BASE + "/toi_{}.html"
THUMBNAIL_URL = URL_BASE + "/toi_{}_files/thumbnail.png"

METADATA_FNAME = "summary.txt"

LINK_HTML = "<a href='{l}'> {txt}</a>"
IMG_HTML = "<img src='{l}' width='100px' height='100px'>"

META_DATA_KEYS = [
    "TOI",
    "TOI html",
    "Thumbnail html",
    "Status",
    "Category",
    "Classification",
    "Runtime [Hr]",
    "Memory [Mb]",
    "Log lines",
]


class TOINotebookMetadata(object):
    """Metadata about a TOI notebook
    - if the notebook exists
    - if the analysis has started
    - if the analysis has completed
    - if the analysis has failed
    - the runtime/memory of the analysis
    """

    def __init__(self, notebook_path: str):
        self.notebook_path = notebook_path
        self.toi: int = grep_toi_number(notebook_path)
        # filepaths (some of these may not exist)
        self.toi_dir = os.path.join(
            os.path.dirname(notebook_path), TOI_DIR.format(toi=self.toi)
        )
        self.tic_data_fname = os.path.join(self.toi_dir, TIC_CSV)
        self.lc_data_fname = os.path.join(self.toi_dir, LC_DATA_FNAME)
        self.inference_data_fname = os.path.join(
            self.toi_dir, INFERENCE_DATA_FNAME
        )
        self.thumbnail_fname = os.path.join(self.toi_dir, THUMBNAIL_PLOT)
        self.profiling_fname = os.path.join(self.toi_dir, PROFILING_CSV)
        self.log_fname = os.path.join(self.toi_dir, LOG_FNAME)
        self.meta_fn = os.path.join(self.toi_dir, METADATA_FNAME)

    @property
    def tic_data(self) -> pd.DataFrame:
        if not hasattr(self, "__tic_data"):
            if os.path.exists(self.tic_data_fname):
                self.__tic_data = pd.read_csv(self.tic_data_fname)
            else:
                self.__tic_data = EXOFOP_DATA.get_tic_data([self.toi])
        return self.__tic_data

    @property
    def analysis_started(self) -> bool:
        required_files_present = all(
            [
                os.path.exists(f)
                for f in [
                    self.notebook_path,
                    self.tic_data_fname,
                    self.lc_data_fname,
                    self.log_fname,
                ]
            ]
        )
        return required_files_present

    @property
    def analysis_completed(self) -> bool:
        required_files_present = all(
            [
                os.path.exists(f)
                for f in [
                    self.inference_data_fname,
                    self.thumbnail_fname,
                ]
            ]
        )
        return required_files_present

    @property
    def analysis_failed(self) -> bool:
        """The analysis has failed if the notebook exists but the inference object has not been saved"""
        return not self.analysis_completed and self.analysis_started

    @property
    def analysis_status(self) -> Status:
        if not self.analysis_started:
            return Status.NOT_STARTED
        elif self.analysis_completed:
            return Status.PASS
        return Status.FAIL

    @property
    def has_multiple_planets(self) -> bool:
        return len(self.tic_data) > 1

    @property
    def toi_category(self) -> str:
        """Get the category of a TOI
        returns: str: "Multi-planet", "Single Transit", "Normal"
        """
        cat = []
        if self.has_multiple_planets:
            cat.append(MULTIPLANET)
        if any(self.tic_data.get(SINGLE_TRANSIT, [])):
            cat.append(SINGLE_TRANSIT)
        if len(cat) == 0:
            cat.append(NORMAL)
        return ", ".join(cat)

    @property
    def classification(self) -> str:
        """Returns classification (eg confirmed-planet, eclipsing binary, etc)"""
        cls_ids = self.tic_data.get("TFOPWG Disposition", [])
        if len(cls_ids) == 0:
            return "U"

        # if cls_ids is a number, convert to string
        if isinstance(cls_ids, (int, float)):
            cls_ids = [str(cls_ids)]

        return ", ".join(cls_ids)

    def __load_profiling_data(self) -> pd.DataFrame:
        """Load the profiling data from the notebook (saved with ploomber)
        Returned dataframe has columns: ['cell', 'runtime', 'memory']
        """
        if os.path.exists(self.profiling_fname):
            return pd.read_csv(self.profiling_fname, index_col=0)
        return pd.DataFrame(columns=["cell", "runtime", "memory"])

    @property
    def profiling_data(self) -> pd.DataFrame:
        if not hasattr(self, "__profiling_data"):
            self.__profiling_data = self.__load_profiling_data()
        return self.__profiling_data

    @property
    def runtime(self) -> float:
        """Time (in Hr) to finish execution"""
        if not hasattr(self, "__runtime"):
            runtime = sum(self.profiling_data.runtime)
            self.__runtime = runtime / (60 * 60)
            if runtime < 1:
                self.__runtime = np.nan
        return self.__runtime

    @property
    def memory(self) -> float:
        """Max RAM used during execution"""
        if not hasattr(self, "__memory"):
            self.__memory = np.nan
            if len(self.profiling_data) > 1:
                self.__memory = max(self.profiling_data.memory)
        return self.__memory

    def get_log_lines(self, n=5) -> str:
        """Get the last n lines of the log file"""
        txt = ""
        if os.path.exists(self.log_fname):
            txt = read_last_n_lines(self.log_fname, n)
        return re.sub(r"(\[(.*?)m)", "", txt)

    @property
    def url(self) -> str:
        """URL to the notebook on catalog website"""
        return NOTEBOOK_URL.format(self.toi)

    @property
    def __toi_html(self) -> str:
        if self.analysis_started:
            return LINK_HTML.format(l=self.url, txt=f"TOI {self.toi}")
        return f"TOI {self.toi}"

    @property
    def __thumbnail_html(self) -> str:
        if self.analysis_completed:
            url = THUMBNAIL_URL.format(self.toi)
            img = IMG_HTML.format(l=url)
            html = LINK_HTML.format(l=self.url, txt=img)
            return html
        return ""

    @property
    def meta_dict(self):
        try:
            meta_dict = {
                "TOI": self.toi,
                "TOI html": self.__toi_html,
                "Thumbnail html": self.__thumbnail_html,
                "Status": self.analysis_status.value,
                "Category": self.toi_category,
                "Classification": self.classification,
                "Runtime [Hr]": self.runtime,
                "Memory [Mb]": self.memory,
                "Log lines": self.get_log_lines(),
            }
        except Exception as e:
            logging.warning(
                f"Failed to get meta data for {self.notebook_path}: {e}"
            )
            meta_dict = {k: np.nan for k in META_DATA_KEYS}
            meta_dict["Log lines"] = f"Metadata extraction failed: {e}"
        if len(set(meta_dict.keys()) - set(META_DATA_KEYS)) > 0:
            raise ValueError(
                f"Invalid metadata keys: {meta_dict.keys()}, expected: {META_DATA_KEYS}"
            )
        return meta_dict

    def save_metadata(self):
        with open(self.meta_fn, "w") as file:
            json.dump(self.meta_dict, file, indent=2)

    def get_meta_data(self) -> Dict[str, Union[str, bool, int, float]]:
        """Returns a dict with metadata about the analysis."""
        if os.path.exists(self.meta_fn):
            with open(self.meta_fn, "r") as file:
                return json.load(file)
        return self.meta_dict
