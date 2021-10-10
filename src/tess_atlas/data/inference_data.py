from pymc3.sampling import MultiTrace
import arviz as az

import logging
import os
import pandas as pd

from tess_atlas.utils import NOTEBOOK_LOGGER_NAME
from .data_object import DataObject

logger = logging.getLogger(NOTEBOOK_LOGGER_NAME)

TRACE_FNAME = "trace.netcdf"


class InferenceData(DataObject):
    def __init__(self, trace):
        self.trace = trace

    @property
    def trace(self) -> az.InferenceData:
        return self._trace

    @trace.setter
    def trace(self, trace):
        if isinstance(trace, MultiTrace):
            self._trace = az.from_pymc3(trace)
        elif isinstance(trace, az.InferenceData):
            self._trace = trace
        else:
            raise TypeError(f"Unknown type: {type(trace)}")

    @classmethod
    def from_cache(cls, outdir: str):
        fname = InferenceData.get_filepath(outdir)
        if not os.path.isfile(fname):
            raise FileNotFoundError(f"{fname} not found.")

        trace = az.from_netcdf(fname)
        logger.info(f"Trace loaded from {fname}")
        return cls(trace=trace)

    def save_data(self, outdir: str):
        fname = self.get_filepath(outdir)
        az.to_netcdf(self.trace, filename=fname)
        logger.info(f"Trace saved at {fname}")

    def get_summary_dataframe(self) -> pd.DataFrame:
        """Returns a dataframe with the mean+sd of each candidate's p, b, r  """
        df = az.summary(
            self.trace,
            var_names=["~lightcurves"],
            filter_vars="like",
        )
        df = (
            df.transpose()
                .filter(regex=r"(.*p\[.*)|(.*r\[.*)|(.*b\[.*)")
                .transpose()
        )
        df = df[["mean", "sd"]]
        df["parameter"] = df.index
        df.set_index(
            ["parameter"], inplace=True, append=False, drop=True
        )
        return df

    @staticmethod
    def get_filepath(outdir):
        return os.path.join(outdir, TRACE_FNAME)
