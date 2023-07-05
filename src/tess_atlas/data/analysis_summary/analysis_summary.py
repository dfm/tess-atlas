import logging
import warnings

import pandas as pd
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map as parallel_tqdm

from tess_atlas.data.exofop import EXOFOP_DATA

from .toi_notebook_metadata import ToiNotebookMetadata

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger("AnalysisSummary")
logger.setLevel(logging.DEBUG)


class AnalysisSummary:
    """Summary of the analysis that has been run.

    This can be used to generate the summary-table of the catalog.
    """

    def __init__(self, summary_data: pd.DataFrame):
        self._data = summary_data
        pass

    def __repr__(self):
        return f"AnalysisSummary(Started[{self.n_analysed}], Pass[{self.n_successful_analyses}] + Failed[{self.n_failed_analyses}] = {self.n_total})"

    @classmethod
    def load_from_outdir(
        self, notebook_dir: str, n_threads=1
    ) -> "AnalysisSummary":
        """Load the metadata from the output directory.

        returns: a pandas DataFrame with the metadata
            Columns:
                TOI, analysis_started, thumbnail, category, classification, runtime,
                analyis_completed, inference_object_saved,

        """
        toi_list = EXOFOP_DATA.get_toi_list(remove_toi_without_lk=False)
        notebook_fns = [f"{notebook_dir}/toi_{toi}.ipynb" for toi in toi_list]
        metadata = [None] * len(notebook_fns)
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
        """Load the metadata from a csv file."""
        return AnalysisSummary(pd.read_csv(csv_path))

    @property
    def dataframe(self):
        """Return a pandas DataFrame with the summary.

        Columns:
        - TOI: the TOI number
        - Classification: the classification of the TOI (confirmed, candidate, etc)
        - Category: the category of the TOI (single, multi, normal)
        - Runtime [Hrs]: the runtime of the analysis in hours
        - Phase Plot: thumbnail of the phase plot
        - Analysis Status: the status of the analysis (Not started, Failed, Completed)

        """
        return self._data.copy()

    def save_to_csv(self, csv_path: str):
        """Save the metadata to a csv file."""
        self._data.to_csv(csv_path, index=False)

    @property
    def n_successful_analyses(self) -> int:
        """Number of TOIs analysed."""
        return sum(self._data.analysis_completed)

    @property
    def n_failed_analyses(self) -> int:
        """Number of TOIs that failed the analysis step."""
        return self.n_analysed - self.n_successful_analyses

    @property
    def n_not_analysed(self) -> int:
        """Number of TOIs that were not analysed."""
        return self.n_total - self.n_analysed

    @property
    def n_analysed(self) -> int:
        """Number of TOIs for which analyses were started."""
        return sum(self._data.analysis_started)

    @property
    def n_total(self) -> int:
        """Total number of TOIs."""
        return len(self._data)


def _get_toi_metadict(fn):
    return ToiNotebookMetadata(fn).__dict__()
