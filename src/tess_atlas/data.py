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
from typing import List

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


@functools.lru_cache()
def get_tic_database():
    return pd.read_csv(TIC_DATASOURCE)


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


class PlanetCandidate:
    """Plant Candidate obtained by TESS."""

    def __init__(
        self,
        toi_id: float,
        period: float,
        t0: float,
        depth: float,
        duration: float,
    ):
        """
        :param float toi_id: The toi number X.Y where the Y represents the
            TOI sub number
        :param float period: Planet candidate orbital period (in days)
        :param float t0: Epoch (timestamp) of the primary transit in
            Barycentric Julian Date
        :param float depth: Planet candidate transit depth, in parts per
            million
        :param float duration: Planet candidate transit duration, in days.
        """
        self.toi_id = toi_id
        self.period = period
        self.t0 = t0
        self.depth = depth
        self.duration = duration

    @classmethod
    def from_toi_database_entry(cls, toi_data: dict):
        return cls(
            toi_id=toi_data["TOI"],
            period=toi_data["Period (days)"],
            t0=toi_data["Epoch (BJD)"] - 2457000,  # convert to TBJD
            depth=toi_data["Depth (ppm)"]
            * 1e-3,  # convert to parts per thousand
            duration=toi_data["Duration (hours)"] / 24.0,  # convert to days
        )

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
        }

    @property
    def is_single_transit(self):
        return (self.period <= 0.0) | np.isnan(self.period)


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


class TICEntry:
    """Hold information about a TIC (TESS Input Catalog) entry"""

    def __init__(self, tic: int, candidates: List[PlanetCandidate], toi: int):
        self.tic_number = tic
        self.toi_number = toi
        self.candidates = candidates
        self.lightcurve = None
        self.setup_outdir()

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
        if tic_data.empty:
            tic_data = get_tic_data_from_database([toi])
        candidates = []
        for index, toi_data in tic_data.iterrows():
            candidate = PlanetCandidate.from_toi_database_entry(
                toi_data.to_dict()
            )
            candidates.append(candidate)
        return cls(
            tic=int(tic_data["TIC ID"].iloc[0]), candidates=candidates, toi=toi
        )

    def load_lightcurve(self):
        self.lightcurve = LightCurveData.from_mast(tic=self.tic_number)
        self.update_missing_candidate_period_guesses_with_lightcurve()

    def update_missing_candidate_period_guesses_with_lightcurve(self):
        generic_period_guess = (
            self.lightcurve.time.max() - self.lightcurve.time.min()
        )
        for planet in self.candidates:
            if planet.is_single_transit:
                planet.period = generic_period_guess

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

    def setup_outdir(self):
        output_dir = os.path.join(f"toi_{self.toi_number}_files")
        os.makedirs(output_dir, exist_ok=True)
        self.outdir = output_dir
