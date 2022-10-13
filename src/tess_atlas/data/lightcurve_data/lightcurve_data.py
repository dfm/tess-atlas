import logging
import os
from typing import List, Optional

import lightkurve as lk
import numpy as np
from lightkurve.lightcurve import TessLightCurve

from tess_atlas.data.data_object import DataObject
from tess_atlas.data.data_utils import residual_rms
from tess_atlas.utils import NOTEBOOK_LOGGER_NAME

from .lightcurve_search import LightcurveSearch

logger = logging.getLogger(NOTEBOOK_LOGGER_NAME)


class LightCurveData(DataObject):
    """Stores Light Curve data for a single target

    Attributes:
        :time np.ndarray: The time in days.
        :flux np.ndarray: The relative flux in parts per thousand.
        :flux_err np.ndarray: The flux err in parts per thousand.

    """

    def __init__(
        self,
        raw_lc: TessLightCurve,
        outdir: str,
    ):
        """
        :raw_lc: sitched raw lk
        :outdir: the outdir to store lk data
        """
        self.raw_lc = raw_lc
        self.cleaned_lc = self.remove_outliers(self.raw_lc)
        formatted_data = self.format_lc_data(self.cleaned_lc)
        self.time = formatted_data["time"]
        self.flux = formatted_data["flux"]
        self.flux_err = formatted_data["flux_err"]
        self.outdir = outdir
        self.len = len(self.time)

    @classmethod
    def from_database(cls, tic: int, outdir: str):
        """Uses lightkurve to get TESS data for a TIC from MAST"""
        logger.info("Downloading LightCurveData from MAST")
        lc = LightcurveSearch(tic).lk_download(outdir)
        logger.info("Completed light curve data download")
        return cls(
            raw_lc=lc,
            outdir=outdir,
        )

    @classmethod
    def from_cache(cls, tic: int, outdir: str):
        fname = LightCurveData.get_filepath(outdir)
        lc = lk.io.tess.read_tess_lightcurve(fname)
        logger.info(f"Lightcurve loaded from {fname}")
        return cls(
            raw_lc=lc,
            outdir=outdir,
        )

    @staticmethod
    def format_lc_data(lc: TessLightCurve):
        t = lc.time.value
        inds = np.argsort(t)
        return dict(
            time=np.ascontiguousarray(t[inds], dtype=np.float64),
            flux=np.ascontiguousarray(
                1e3 * (lc.flux.value[inds] - 1), dtype=np.float64
            ),
            flux_err=np.ascontiguousarray(
                1e3 * lc.flux_err.value[inds], dtype=np.float64
            ),
        )

    @staticmethod
    def remove_outliers(
        lc: TessLightCurve, window_length=11, sigma=100, rms_threshold=5
    ) -> TessLightCurve:
        """Removes outliers far away from lc trend

        Note: Trend obtained by removeing low frequency info using scipy’s Savitzky-Golay filter.
        Larger the window -- more "smooth" the data

        :param lc: the lightcurve which will have outliers removed
        :type lc: TessLightCurve
        :param window_length: +ive odd int
        :type window_length: int
        :param sigma: how far away from the "trend" should things be clipped
        :type sigma: float
        :param rms_threshold: the tolerance for the lc-tred residual (the outlier check)
        :type rms_threshold: float
        """
        _, trend = lc.flatten(
            return_trend=True, window_length=window_length, sigma=sigma
        )
        resid = lc.flux - trend.flux
        rms = residual_rms(resid)
        good = resid < rms_threshold * rms
        return lc[good]

    @property
    def cadence(self):
        """How often TESS photometric observations are stored."""
        return np.min(np.diff(self.time))

    def save_data(self, outdir):
        fpath = self.get_filepath(outdir)
        self.raw_lc.to_fits(fpath, overwrite=True)

    @staticmethod
    def get_filepath(outdir, fname="lightkurve_lc.fits"):
        return os.path.join(outdir, fname)

    def get_observation_durations(self, tic):
        data = LightcurveSearch(tic).lk_download(self.outdir)
        observation_durations = []
        for obs in data:
            t = obs.time.value
            observation_durations.append([min(t), max(t)])
        return np.array(observation_durations)

    def _repr_html_(self):
        return (
            f"{self.__class__.__name__}: "
            f"{self.len} datapoints "
            # f"({self.mem_size})"
        )

    def timefold(self, t0, p):
        hp = 0.5 * p
        return (self.time - t0 + hp) % p - hp

    def filter_non_transit_data(
        self,
        candidates: List,
    ):
        """Remove data outside transits (keep 'day-buffer' days near transit)"""

        max_duration = max([c.duration * 2 for c in candidates])
        day_buffer = max(0.3, max_duration)

        transit_mask = self.get_transit_mask(candidates, day_buffer)
        new_len = sum(transit_mask)
        logger.info(
            f"Reducing lightcurve from {self.len:,}-->{new_len:,} [day-buffer = {day_buffer:.1f}]"
        )
        self.cleaned_lc = self.cleaned_lc[transit_mask]
        formatted_data = self.format_lc_data(self.cleaned_lc)
        self.time = formatted_data["time"]
        self.flux = formatted_data["flux"]
        self.flux_err = formatted_data["flux_err"]
        self.len = len(self.time)

    def get_transit_mask(
        self,
        candidates: List,
        day_buffer: Optional[float] = 0.3,
    ):
        """Get mask for data with transits (keep 'day-buffer' days near transit)"""
        transit_mask = np.zeros(self.len).astype(bool)
        for p in candidates:
            transit_mask |= (
                np.abs(self.timefold(p.tmin, p.period)) < day_buffer
            )
        return transit_mask


def check_transit_mask(tic_entry):
    import matplotlib.pyplot as plt

    candidates = tic_entry.candidates
    max_duration = max([c.duration * 2 for c in candidates])
    day_buffer = max(0.3, max_duration)
    transit_mask = tic_entry.lightcurve.get_transit_mask(
        candidates, day_buffer
    )
    t = tic_entry.lightcurve.time
    y = tic_entry.lightcurve.flux
    plt.plot(t, y)
    plt.plot(t, transit_mask * min(y))
