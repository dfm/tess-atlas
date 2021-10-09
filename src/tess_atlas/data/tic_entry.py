import functools
import json
import logging
import os
from typing import Dict, List, Optional

import arviz as az
import pandas as pd
from IPython.display import display
from IPython.display import HTML
from pymc3.sampling import MultiTrace

from .data_object import DataObject

from tess_atlas.utils import NOTEBOOK_LOGGER_NAME
from .lightcurve_data import LightCurveData
from .planet_candidate import PlanetCandidate
from .stellar_data import StellarData

logger = logging.getLogger(NOTEBOOK_LOGGER_NAME)

EXOFOP = "https://exofop.ipac.caltech.edu/tess/"
TIC_DATASOURCE = EXOFOP + "download_toi.php?sort=toi&output=csv"
TIC_SEARCH = EXOFOP + "target.php?id={tic_id}"

TOI_DIR = "toi_{toi}_files"

DIR = os.path.dirname(__file__)


TIC_FNAME = "tic_data.csv"

def get_tic_database():
    # if we have a cached database file
    cached_file = os.path.join(DIR, "cached_tic_database.csv")
    if os.path.isfile(cached_file):
        return pd.read_csv(cached_file)

    # go online to grab database and cache
    db = pd.read_csv(TIC_DATASOURCE)
    db.to_csv(cached_file)
    return db


def get_tic_id_for_toi(toi_number: int) -> int:
    tic_db = get_tic_database()
    toi = tic_db[tic_db["TOI"] == toi_number + 0.01].iloc[0]
    return int(toi["TIC ID"])


def get_tic_data_from_database(toi_numbers: List[int]) -> pd.DataFrame:
    """Get rows of about a TIC  from ExoFOP associated with a TOI target.
    :param int toi_numbers: The list TOI number for which the TIC data is required
    :return: Dataframe with all TOIs for the TIC which contains TOI {toi_id}
    :rtype: pd.DataFrame
    """
    tic_db = get_tic_database()
    tics = [get_tic_id_for_toi(toi) for toi in toi_numbers]
    dfs = [tic_db[tic_db["TIC ID"] == tic].sort_values("TOI") for tic in tics]
    tois_for_tic = pd.concat(dfs)
    if len(tois_for_tic) < 1:
        raise ValueError(f"TOI data for TICs-{tics} does not exist.")
    return tois_for_tic


class TICEntry(DataObject):
    """Hold information about a TIC (TESS Input Catalog) entry"""

    def __init__(
            self,
            tic_number: int,
            toi: int,
            lightcurve: LightCurveData,
            stellar_data: StellarData,
            tic_data: pd.DataFrame,
            loaded_from_cache:Optional[bool]=False
    ):
        self.tic_number = tic_number
        self.toi_number = toi
        self.lightcurve = lightcurve
        self.stellar_data = stellar_data
        self.tic_data = tic_data
        self.candidates = self.get_candidates()
        self.outdir = TOI_DIR.format(toi=toi)
        self.loaded_from_cache = loaded_from_cache

    def get_candidates(self):
        candidates = []
        for index, toi_data in self.tic_data.iterrows():
            candidate = PlanetCandidate.from_database(
                toi_data=toi_data.to_dict(),
                lightcurve=self.lightcurve
            )
            candidates.append(candidate)
        return candidates

    @property
    def exofop_url(self):
        return TIC_SEARCH.format(tic_id=self.tic_number)

    @property
    def planet_count(self):
        return len(self.candidates)

    @property
    def inference_trace(self) -> az.InferenceData:
        return self._inference_trace

    @inference_trace.setter
    def inference_trace(self, inference_trace):
        if isinstance(inference_trace, MultiTrace):
            self._inference_trace = az.from_pymc3(inference_trace)
        elif isinstance(inference_trace, az.InferenceData):
            self._inference_trace = inference_trace
        else:
            raise TypeError(f"Unknown type: {type(inference_trace)}")

    def load_inference_trace(self, fname=None):
        if fname is None:
            fname = self.inference_trace_filename
        if os.path.isfile(fname):
            logger.info(f"Trace loaded from {self.inference_trace_filename}")
            self.inference_trace = az.from_netcdf(
                self.inference_trace_filename
            )
        else:
            raise FileNotFoundError(f"{fname} not found.")

    def save_inference_trace(self, fname=None):
        if fname is None:
            fname = self.inference_trace_filename
        az.to_netcdf(self.inference_trace, filename=fname)
        logger.info(f"Trace saved at {fname}")

    @classmethod
    def load_tic_data(
            cls, toi: int, tic_data: pd.DataFrame = pd.DataFrame()
    ):
        toi_dir = TOI_DIR.format(toi=toi)
        if os.path.isdir(toi_dir):
            return cls.from_cache(toi, toi_dir)
        else:
            return cls.from_database(toi)

    @classmethod
    def from_database(cls, toi: int):
        tic_data = get_tic_data_from_database([toi])
        tic_number = int(tic_data["TIC ID"].iloc[0])
        return cls(
            tic_number=tic_number,
            toi=toi,
            lightcurve=LightCurveData.from_database(tic=tic_number),
            stellar_data=StellarData.from_database(tic=tic_number),
            tic_data=tic_data
        )

    @classmethod
    def from_cache(cls, toi:int, outdir:str):
        tic_data = pd.read_csv(
            os.path.join(outdir, TIC_FNAME)
        )
        tic_number = int(tic_data["TIC ID"].iloc[0])
        return cls(
            tic_number=tic_number,
            toi=toi,
            lightcurve=LightCurveData.from_cache(outdir),
            stellar_data=StellarData.from_cache(outdir),
            tic_data=tic_data,
            loaded_from_cache=True
        )

    def to_dataframe(self):
        return pd.DataFrame(
            [candidate.to_dict() for candidate in self.candidates]
        )

    def display(self):
        df = self.to_dataframe()
        df = df.transpose()
        df.columns = df.loc["TOI"]
        display(df)
        more_data_str = "More data on ExoFOP page"
        html_str = f"<a href='{self.exofop_url}'>{more_data_str}</a>"
        display(HTML(html_str))

    @property
    def inference_trace_filename(self):
        return os.path.join(self.outdir, f"toi_{self.toi_number}.netcdf")

    def get_trace_summary(self) -> pd.DataFrame:
        """Returns a dataframe with the mean+sd of each candidate's p, b, r  """
        df = az.summary(
            self.inference_trace,
            var_names=["~lightcurves"],
            filter_vars="like",
        )
        df = (
            df.transpose()
                .filter(regex=r"(.*p\[.*)|(.*r\[.*)|(.*b\[.*)")
                .transpose()
        )
        df = df[["mean", "sd"]]
        df["TOI"] = self.toi_number
        df["parameter"] = df.index
        df.set_index(
            ["TOI", "parameter"], inplace=True, append=False, drop=True
        )
        return df

    @property
    def outdir(self):
        return self.__outdir

    @outdir.setter
    def outdir(self, outdir):
        os.makedirs(outdir, exist_ok=True)
        self.__outdir = outdir
        self.save_data()

    def save_data(self):
        self.tic_data.to_csv(
            os.path.join(self.outdir, TIC_FNAME)
        )
        self.lightcurve.save_data(self.outdir)
        self.stellar_data.save_data(self.outdir)

