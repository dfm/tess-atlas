import logging
import warnings
from typing import Dict, List, Union

import pandas as pd
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map as parallel_tqdm

from ..notebook_controllers.controllers.toi_notebook_controller import (
    Status,
    TOINotebookController,
)
from .exofop import EXOFOP_DATA

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger("AnalysisSummary")
logger.setLevel(logging.DEBUG)


class AnalysisSummary:
    """Summary of the analysis that has been run.

    This can be used to generate the summary-table of the catalog.
    """

    def __init__(self, summary_data: pd.DataFrame):
        self._data = summary_data
        self._set_counts()

    def _set_counts(self):
        counts = self._data["Status"].value_counts().to_dict()
        self.n_successful_analyses = counts.get(Status.PASS.value, 0)
        self.n_failed_analyses = counts.get(Status.FAIL.value, 0)
        self.n_not_analysed = counts.get(Status.NOT_STARTED.value, 0)
        self.n_analysed = self.n_successful_analyses + self.n_failed_analyses
        self.n_total = len(self._data)

    def __repr__(self):
        return f"AnalysisSummary(Started[{self.n_analysed}], Pass[{self.n_successful_analyses}] + Failed[{self.n_failed_analyses}] = {self.n_total})"

    @classmethod
    def from_dir(self, notebook_dir: str, n_threads=1) -> "AnalysisSummary":
        """Load the metadata from the output directory.
        returns: a pandas DataFrame with the metadata
        """
        toi_list = EXOFOP_DATA.get_toi_list(remove_toi_without_lk=True)
        notebook_fns = [f"{notebook_dir}/toi_{toi}.ipynb" for toi in toi_list]
        metadata: List[Union[Dict, None]] = [None] * len(notebook_fns)
        kwrgs = dict(desc="Parsing notebook metadata", total=len(toi_list))
        if n_threads == 1:
            for i, fn in tqdm(enumerate(notebook_fns), **kwrgs):
                metadata[i] = _get_toi_metadict(fn)
        else:
            metadata = parallel_tqdm(
                _get_toi_metadict,
                notebook_fns,
                **kwrgs,
            )
        df = pd.DataFrame(metadata)
        return AnalysisSummary(df)

    @classmethod
    def load_from_csv(self, csv_path: str) -> "AnalysisSummary":
        return AnalysisSummary(pd.read_csv(csv_path))

    @property
    def dataframe(self):
        """Return a pandas DataFrame with the summary.

        Columns:
        - TOI: the TOI number
        - Thumbnail html: the html to embed the thumbnail of the TOI
        - TOI html: the html link to the TOI
        - Classification: the classification of the TOI (confirmed, candidate, etc)
        - Category: the category of the TOI (single, multi, normal)
        - Runtime [Hr]: the runtime of the analysis in hours
        - Memory [Mb]: the memory usage of the analysis in MB
        - Status: the status of the analysis (Not started, Failed, Completed)
        """
        return self._data.copy()

    def save_to_csv(self, csv_path: str):
        self._data.to_csv(csv_path, index=False)
        return csv_path


def _get_toi_metadict(fn: str) -> Dict[str, Union[str, bool, int, float]]:
    return TOINotebookController(fn).get_meta_data()
