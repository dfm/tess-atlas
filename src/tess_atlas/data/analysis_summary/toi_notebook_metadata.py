"""Module to parse the metadata from the TOI notebooks"""
import json
import os

import numpy as np
import pandas as pd
from strenum import StrEnum

from tess_atlas.data.exofop.constants import (
    MULTIPLANET,
    NORMAL,
    SINGLE_TRANSIT,
)

from ... import __website__
from ...file_management import (
    INFERENCE_DATA_FNAME,
    PROFILING_CSV,
    TIC_CSV,
    TOI_DIR,
    read_last_n_lines,
)
from ...logger import LOG_FNAME
from ...plotting.labels import THUMBNAIL_PLOT
from ...utils import grep_toi_number
from ..planet_candidate import CLASS_SHORTHAND

URL_BASE = f"{__website__}/content/toi_notebooks"
NOTEBOOK_URL = URL_BASE + "/toi_{}.html"
THUMBNAIL_URL = URL_BASE + "/toi_{}_files/thumbnail.png"


class Status(StrEnum):
    PASS = "completed"
    FAIL = "failed"
    NOT_STARTED = "not started"


METADATA_FNAME = "summary.txt"


class ToiNotebookMetadata(object):
    """Metadata about a TOI notebook
    - if the notebook exists
    - if the analysis has started
    - if the analysis has completed
    - if the analysis has failed
    - if the inference object has been saved
    - the runtime of the analysis

    #TODO: merge with the toi_notebook_controller class? or a mixin?
    """

    def __init__(self, path: str):
        self.path = path
        self.toi: int = grep_toi_number(path)
        # filepaths (some of these may not exist)
        self.toi_dir = os.path.join(
            os.path.dirname(path), TOI_DIR.format(toi=self.toi)
        )
        self.tic_data_fname = os.path.join(self.toi_dir, TIC_CSV)
        self.inference_object_fname = os.path.join(
            self.toi_dir, INFERENCE_DATA_FNAME
        )
        self.thumbnail_fname = os.path.join(self.toi_dir, THUMBNAIL_PLOT)
        self.profiling_fname = os.path.join(self.toi_dir, PROFILING_CSV)
        self.log_fname = os.path.join(self.toi_dir, LOG_FNAME)
        self.save_fn = os.path.join(self.toi_dir, METADATA_FNAME)

    @property
    def tic_data(self) -> pd.DataFrame:
        if not hasattr(self, "__tic_data"):
            self.__tic_data = pd.DataFrame()
            if os.path.exists(self.tic_data_fname):
                self.__tic_data = pd.read_csv(self.tic_data_fname)
        return self.__tic_data

    @property
    def notebook_exists(self) -> bool:
        return os.path.exists(self.path)

    @property
    def inference_object_saved(self) -> bool:
        return os.path.exists(self.inference_object_fname)

    @property
    def analysis_started(self) -> bool:
        return self.notebook_exists

    @property
    def analysis_completed(self) -> bool:
        return (
            os.path.exists(self.thumbnail_fname)
            and self.inference_object_saved
        )

    @property
    def analysis_failed(self) -> bool:
        """The analysis has failed if the notebook exists but the inference object has not been saved"""
        return not self.analysis_completed and self.analysis_started

    @property
    def analysis_status(self) -> Status:
        """The status of the analysis: NOT_STARTED, PASS, FAIL"""
        if not self.analysis_started:
            return Status.NOT_STARTED
        elif self.analysis_completed:
            return Status.PASS
        else:
            return Status.FAIL

    @property
    def has_multiple_planets(self) -> bool:
        return len(self.tic_data) > 1

    @property
    def toi_category(self) -> str:
        """Get the category of a TOI
        returns:
            str: "Multi-planet", "Single Transit", "Normal"
        """

        cat = []
        if self.has_multiple_planets:
            cat.append(MULTIPLANET)

        if len(self.tic_data[self.tic_data[SINGLE_TRANSIT]]) > 1:
            cat.append(SINGLE_TRANSIT)

        if len(cat) == 0:
            cat.append(NORMAL)

        return ", ".join(cat)

    @property
    def classification(self) -> str:
        """Returns classification (eg confirmed-planet, eclipsing binary, etc)"""
        cls_ids = self.tic_data["TESS Disposition"].values
        cls_ids = [CLASS_SHORTHAND.get(_id, _id) for _id in cls_ids]
        return ", ".join(cls_ids)

    def load_profiling_data(self) -> pd.DataFrame:
        """Load the profiling data from the notebook (saved with ploomber)
        Returned dataframe has columns: ['cell', 'runtime', 'memory']
        """
        if os.path.exists(self.profiling_fname):
            return pd.read_csv(self.profiling_fname, index_col=0)
        return pd.DataFrame(columns=["cell", "runtime", "memory"])

    @property
    def profiling_data(self) -> pd.DataFrame:
        if not hasattr(self, "__profiling_data"):
            self.__profiling_data = self.load_profiling_data()
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
        if os.path.exists(self.log_fname):
            return read_last_n_lines(self.log_fname, n)
        return "LOGS NOT FOUND"

    @property
    def url(self) -> str:
        """URL to the notebook on catalog website"""
        return NOTEBOOK_URL.format(self.toi)

    @property
    def toi_html(self) -> str:
        """HTML link to the notebook on catalog website"""
        if self.notebook_exists:
            return f"<a href='{self.url}'>TOI {self.toi}</a>"
        else:
            return f"TOI {self.toi}"

    @property
    def thumbnail_html(self) -> str:
        """HTML code to embedthe thumbnail on catalog website"""
        if self.analysis_completed:
            url = THUMBNAIL_URL.format(self.toi)
            return f"<a href='{self.url}'> <img src='{url}' width='200' height='200'> </a>"
        else:
            return ""

    def __repr__(self):
        return f"<TOI{self.toi} Notebook>"

    def __dict__(self):
        return {
            "TOI": self.toi,
            "TOI html": self.toi_html,
            "Thumbnail": self.thumbnail_html,
            "Status": self.analysis_status.value,
            "Category": self.toi_category,
            "Classification": self.classification,
            "Runtime [Hr]": self.runtime,
            "Memory [Mb]": self.inference_object_saved,
            "Log lines": self.get_log_lines(),
        }

    def save(self):
        with open(self.save_fn, "w") as file:
            json.dump(self.__dict__(), file, indent=2)

    def get_meta_data(self):
        if os.path.exists(self.save_fn):
            with open(self.save_fn, "r") as file:
                return json.load(file)
        return self.__dict__()
