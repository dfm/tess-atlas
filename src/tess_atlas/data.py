# -*- coding: utf-8 -*-

__all__ = [
    "get_tic_database",
    "get_tic_id_for_toi",
    "get_tic_data_from_database",
    "calculate_time_fold",
    "PlanetCandidate",
    "LightCurveData",
    "TICEntry",
]

import functools
import logging
import os
from typing import List, Dict, Optional
import json

import arviz as az
import lightkurve as lk
import numpy as np
import pandas as pd
from IPython.display import display
from pymc3.sampling import MultiTrace

logging.getLogger().setLevel(logging.INFO)

TIC_DATASOURCE = (
    "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
)

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


def calculate_time_fold(t, t0, p):
    """Function to get time-fold"""
    hp = 0.5 * p
    return (t - t0 + hp) % p - hp


class LightCurveData:
    """Stores Light Curve data for a single target"""

    def __init__(
            self, time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray
    ):
        """
        :param np.ndarray time: The time in days.
        :param np.ndarray flux: The relative flux in parts per thousand.
        :param np.ndarray fluex_err: The flux err in parts per thousand.
        """
        self.time = time
        self.flux = flux
        self.flux_err = flux_err

    @classmethod
    def from_mast(cls, tic: int):
        """Uses lightkurve to get TESS data for a TIC from MAST"""
        logging.info(
            f"Searching for lightkurve data with target='TIC {tic}', "
            "mission='TESS'"
        )
        search = lk.search_lightcurve(target=f"TIC {tic}", mission="TESS")
        logging.debug(f"Search  succeeded: {search}")

        # Restrict to short cadence no "fast" cadence
        search = search[np.where(search.table["t_exptime"] == 120)]

        logging.info(
            f"Downloading {len(search)} observations of light curve data "
            f"(TIC {tic})"
        )
        data = search.download_all()
        if data is None:
            raise ValueError(f"No light curves for TIC {tic}")
        logging.info("Completed light curve data download")
        data = data.stitch()
        data = data.remove_nans().remove_outliers(sigma=7)
        t = data.time.value
        inds = np.argsort(t)
        return cls(
            time=np.ascontiguousarray(t[inds], dtype=np.float64),
            flux=np.ascontiguousarray(
                1e3 * (data.flux.value[inds] - 1), dtype=np.float64
            ),
            flux_err=np.ascontiguousarray(
                1e3 * data.flux_err.value[inds], dtype=np.float64
            ),
        )

    def to_dict(self):
        return {
            "time": self.time,
            "flux": self.flux,
            "flux_err": self.flux_err,
        }

    def save_data(self, outdir):
        pd.DataFrame(self.to_dict()).to_csv(os.path.join(outdir, "lightcurve.csv"), index=False)
        logging.info(f"Saved lightcurve data.")

class PlanetCandidate:
    """Plant Candidate obtained by TESS."""

    def __init__(
            self,
            toi_id: float,
            period: float,
            time: np.ndarray,
            t0: float,
            depth: float,
            duration: float,
    ):
        """
        :param float toi_id: The toi number X.Y where the Y represents the
            TOI sub number (e.g. 103.1)
        :param np.ndarray time: The list of times for which we have data for
        :param float period: Planet candidate orbital period (in days)
        :param float t0: Epoch (timestamp) of the primary transit in
            Barycentric Julian Date
        :param float depth: Planet candidate transit depth, in parts per
            million
        :param float duration: Planet candidate transit duration, in days.
        """
        self.toi_id = toi_id
        self.t0 = t0
        self.period = period
        self.depth = depth
        self.duration = duration
        self.__time = time

    @property
    def index(self):
        return int(str(self.toi_id).split(".")[1])

    @property
    def num_periods(self):
        """number of periods between t0 and tmax"""
        return (np.floor(max(self.__time) - self.t0) / self.period).astype(int)

    @property
    def tmax(self):
        """Time of the last transit"""
        return self.t0 + self.num_periods * self.period

    @property
    def period_min(self):
        """the minimum possible period"""
        return np.maximum(
            np.abs(self.t0 - self.__time.max()),
            np.abs(self.__time.min() - self.t0)
        )

    @property
    def duration_max(self):
        if self.has_data_only_for_single_transit:
            return 1.0
        return max(1.5 * self.duration, 0.1)

    @property
    def duration_min(self):
        return  min(self.duration, 2 * np.min(np.diff(self.__time)))

    @property
    def has_data_only_for_single_transit(self):
        return (self.period <= 0.0) or np.isnan(self.period)

    @classmethod
    def from_toi_database_entry(cls, toi_data: Dict, lightcurve: LightCurveData):
        unpack_data = dict(
            toi_id=toi_data["TOI"],
            period=toi_data["Period (days)"],
            t0=toi_data["Epoch (BJD)"] - 2457000,  # convert to TBJD
            depth=toi_data["Depth (ppm)"] * 1e-3,  # convert to parts per thousand
            duration=toi_data["Duration (hours)"] / 24.0,  # convert to days,
            time=lightcurve.time
        )
        return cls(**unpack_data)

    def get_timefold(self, t):
        """Used in plotting"""
        return calculate_time_fold(t, self.t0, self.period)

    def to_dict(self):
        return {
            "TOI": self.toi_id,
            "Period (days)": self.period,
            "Epoch (TBJD)": self.t0,
            "Depth (ppt)": self.depth,
            "Duration (days)": self.duration,
            "Single Transit": self.has_data_only_for_single_transit,
        }



class TICEntry:
    """Hold information about a TIC (TESS Input Catalog) entry"""

    def __init__(self, tic_number: int, candidates: List[PlanetCandidate], toi: int, lightcurve: LightCurveData,
                 meta_data: Optional[Dict] = {}):
        self.tic_number = tic_number
        self.toi_number = toi
        self.candidates = candidates
        self.lightcurve = lightcurve
        self.meta_data = meta_data
        self.outdir = os.path.join(f"toi_{self.toi_number}_files")

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
            logging.debug(f"Trace loaded from {self.inference_trace_filename}")
            self.inference_trace = az.from_netcdf(
                self.inference_trace_filename
            )
        else:
            raise FileNotFoundError(f"{fname} not found.")

    def save_inference_trace(self, fname=None):
        if fname is None:
            fname = self.inference_trace_filename
        az.to_netcdf(self.inference_trace, filename=fname)
        logging.info(f"Trace saved at {fname}")

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
            tic_number=tic_number, candidates=candidates, toi=toi, lightcurve=lightcurve, meta_data=tic_data.to_dict('list')
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
