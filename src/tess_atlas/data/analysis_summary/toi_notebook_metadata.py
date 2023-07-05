"""Module to parse the metadata from the TOI notebooks"""
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from strenum import StrEnum

from tess_atlas.data.exofop import EXOFOP_DATA
from tess_atlas.data.exofop.constants import (
    MULTIPLANET,
    NORMAL,
    SINGLE_TRANSIT,
)
from tess_atlas.data.inference_data_tools import INFERENCE_DATA_FNAME
from tess_atlas.data.planet_candidate import CLASS_SHORTHAND
from tess_atlas.plotting.labels import THUMBNAIL_PLOT
from tess_atlas.utils import grep_toi_number

URL_BASE = "http://catalog.tess-atlas.cloud.edu.au/content/toi_notebooks"
NOTEBOOK_URL = URL_BASE + "/toi_{}.html"


class Status(StrEnum):
    PASS = ("completed",)
    FAIL = ("failed",)
    NOT_STARTED = "not started"


class ToiNotebookMetadata(object):
    """Metadata about a TOI notebook

    - if the notebook exists
    - if the analysis has started
    - if the analysis has completed
    - if the analysis has failed
    - if the inference object has been saved
    - the runtime of the analysis
    """

    def __init__(self, path: str):
        self.path = path
        self.toi: int = grep_toi_number(path)
        self.tic_data = self._get_tic_data()
        self.category = self.get_toi_category()
        self.profiling_data: pd.DataFrame = None

    def _get_tic_data(self):
        # TODO: does this work for multiplanet systems?
        return EXOFOP_DATA.get_tic_data([self.toi]).to_dict("records")[0]

    @property
    def notebook_exists(self) -> bool:
        return os.path.exists(self.path)

    @property
    def basedir(self) -> str:
        return os.path.dirname(self.path)

    @property
    def datadir(self) -> str:
        return os.path.join(self.basedir, f"toi_{self.toi}_files")

    @property
    def thumbnail(self) -> str:
        """TODO: make a thumbnail with all the phase plots in it (for multiplanet systems)"""
        return os.path.join(self.datadir, THUMBNAIL_PLOT)

    @property
    def inference_object_saved(self) -> bool:
        f = os.path.join(self.datadir, INFERENCE_DATA_FNAME)
        return os.path.exists(f)

    @property
    def analysis_started(self) -> bool:
        """The analysis has started if the notebook exists"""
        return self.notebook_exists

    @property
    def analyis_completed(self) -> bool:
        """The analysis has completed if the inference object has been saved + the thumbnail exists"""
        return os.path.exists(self.thumbnail)

    @property
    def analysis_failed(self) -> bool:
        """The analysis has failed if the notebook exists but the inference object has not been saved"""
        return not self.analyis_completed and self.analysis_started

    @property
    def analysis_status(self) -> Status:
        """The status of the analysis: NOT_STARTED, PASS, FAIL"""
        if not self.analysis_started:
            return Status.NOT_STARTED
        elif self.analyis_completed:
            return Status.PASS
        else:
            return Status.FAIL

    def get_toi_category(self) -> str:
        """Get the category of a TOI
        returns:
            str: "Multi-planet", "Single Transit", "Normal"
        """
        if self.tic_data[MULTIPLANET]:
            cat = MULTIPLANET
            if self.tic_data[SINGLE_TRANSIT]:
                cat += f", {SINGLE_TRANSIT}"
        elif self.tic_data[SINGLE_TRANSIT]:
            cat = SINGLE_TRANSIT
        else:
            cat = NORMAL
        return cat

    @property
    def classification(self) -> str:
        """Returns classification (eg confirmed-planet, eclipsing binary, etc)"""
        classid = self.tic_data["TESS Disposition"]
        return CLASS_SHORTHAND.get(classid, classid)

    def load_profiling_data(self) -> pd.DataFrame:
        """Load the profiling data from the notebook (saved with ploomber)"""
        # TODO
        raise NotImplementedError

    @property
    def runtime(self) -> float:
        """Time (in Hr) to finish execution"""
        return np.nan  # TODO: read in from CSV

    @property
    def logfile(self) -> str:
        """Path to the logfile"""
        raise NotImplementedError

    def get_log_lines(self, n=10) -> str:
        """Returns the last n lines from the logs"""
        raise NotImplementedError

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
        if self.analyis_completed:
            return f"<a href='{self.url}'> <img src='{self.thumbnail}' width='200' height='200'> </a>"
        else:
            return ""

    def __repr__(self):
        return f"<TOI{self.toi} Notebook>"

    def __dict__(self):
        return {
            "TOI": self.toi_html,
            "analysis_started": self.analysis_started,
            "analysis_completed": self.analyis_completed,
            "analysis_status": self.analysis_status.value,
            "thumbnail": self.thumbnail_html,
            "category": self.category,
            "classification": self.classification,
            "runtime": self.runtime,
            "inference_object_saved": self.inference_object_saved,
            **self.tic_data,
        }
