from pymc3.sampling import MultiTrace
import arviz as az

import logging
import os
import pandas as pd

from tess_atlas.utils import NOTEBOOK_LOGGER_NAME
from .data_object import DataObject

logger = logging.getLogger(NOTEBOOK_LOGGER_NAME)

SAMPLES_FNAME = "samples.csv"


class InferenceData(DataObject):
    def __init__(self, trace):
        self.trace = trace
        self.samples = self.get_samples_dataframe()

    @property
    def trace(self) -> az.InferenceData:
        return self._trace

    @trace.setter
    def trace(self, trace):
        """Makes sure that the trace is a `az.InferenceData` object"""
        if isinstance(trace, MultiTrace):
            self._trace = az.from_pymc3(trace)
        elif isinstance(trace, az.InferenceData):
            self._trace = trace
        else:
            raise TypeError(f"Unknown type: {type(trace)}")

    @classmethod
    def load(cls, outdir: str):
        fname = InferenceData.get_filepath(outdir)
        if not os.path.isfile(fname):
            raise FileNotFoundError(f"{fname} not found.")

        trace = az.from_netcdf(fname)
        logger.info(f"Inference trace loaded from {fname}")
        return cls(trace=trace)

    def save_data(self, outdir: str):
        fname = self.get_filepath(outdir)
        az.to_netcdf(self.trace, filename=fname)
        self.save_samples(outdir)
        logger.info(f"Trace saved at {fname}")

    def get_summary_dataframe(self) -> pd.DataFrame:
        """Returns a dataframe with the mean+sd of each candidate's p, b, r"""
        df = az.summary(
            self.trace, var_names=["~lightcurves"], filter_vars="like"
        )
        df = (
            df.transpose()
            .filter(regex=r"(.*p\[.*)|(.*r\[.*)|(.*b\[.*)")
            .transpose()
        )
        df = df[["mean", "sd"]]
        df["parameter"] = df.index
        df.set_index(["parameter"], inplace=True, append=False, drop=True)
        return df

    @staticmethod
    def get_filepath(outdir, fname="trace.netcdf"):
        return os.path.join(outdir, fname)

    def get_samples_dataframe(self) -> pd.DataFrame:
        df = self.trace.to_dataframe(groups=["posterior"])
        samples = df.filter(regex=r"(.*p\[.*)|(.*r\[.*)|(.*b\[.*)")
        # samples cols are like: [('r[0]', 0), ('b[0]', 0), ('p[0]', 0), ...
        new_cols = []
        for (posterior_label, group) in list(samples.columns):
            # converting column labels to only include posterior label
            new_cols.append(posterior_label)
        samples.columns = new_cols
        return samples

    def save_samples(self, outdir):
        fpath = os.path.join(outdir, SAMPLES_FNAME)
        self.samples.to_csv(fpath, index=False)
