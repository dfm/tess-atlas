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

from tess_atlas.utils import NOTEBOOK_LOGGER_NAME
from .lightcurve_data import LightCurveData
from .planet_candidate import PlanetCandidate

logger = logging.getLogger(NOTEBOOK_LOGGER_NAME)

EXOFOP = "https://exofop.ipac.caltech.edu/tess/"
TIC_DATASOURCE = EXOFOP + "download_toi.php?sort=toi&output=csv"
TIC_SEARCH = EXOFOP + "target.php?id={tic_id}"

DIR = os.path.dirname(__file__)


@functools.lru_cache()
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


class TICEntry:
    """Hold information about a TIC (TESS Input Catalog) entry"""

    def __init__(
        self,
        tic_number: int,
        candidates: List[PlanetCandidate],
        toi: int,
        lightcurve: LightCurveData,
        meta_data: Optional[Dict] = {},
    ):
        self.tic_number = tic_number
        self.toi_number = toi
        self.candidates = candidates
        self.lightcurve = lightcurve
        self.meta_data = meta_data
        self.meta_data["exofop_url"] = self.exofop_url
        self.outdir = os.path.join(f"toi_{self.toi_number}_files")

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
    def generate_tic_from_toi_number(
        cls, toi: int, tic_data: pd.DataFrame = pd.DataFrame()
    ):
        # Load database entry for TIC
        tic_data = get_tic_data_from_database([toi])
        tic_number = int(tic_data["TIC ID"].iloc[0])

        # Load lighcurve for TIC
        lightcurve = LightCurveData.from_mast(tic=tic_number)

        candidates = []
        for index, toi_data in tic_data.iterrows():
            candidate = PlanetCandidate.from_toi_database_entry(
                toi_data=toi_data.to_dict(), lightcurve=lightcurve
            )
            candidates.append(candidate)

        return cls(
            tic_number=tic_number,
            candidates=candidates,
            toi=toi,
            lightcurve=lightcurve,
            meta_data=tic_data.to_dict("list"),
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
        self.cache_data()

    def cache_data(self):
        with open(os.path.join(self.outdir, "meta_data.json"), "w") as outfile:
            json.dump(self.meta_data, outfile, indent=4, sort_keys=True)
        self.lightcurve.save_data(self.outdir)
