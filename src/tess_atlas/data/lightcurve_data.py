import logging
import os

import lightkurve as lk
import numpy as np
import pandas as pd

from tess_atlas.utils import NOTEBOOK_LOGGER_NAME

from .data_object import DataObject
from ..utils import get_cache_dir


logger = logging.getLogger(NOTEBOOK_LOGGER_NAME)


class LightCurveData(DataObject):
    """Stores Light Curve data for a single target"""

    def __init__(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: np.ndarray,
        outdir: str,
    ):
        """
        :param np.ndarray time: The time in days.
        :param np.ndarray flux: The relative flux in parts per thousand.
        :param np.ndarray fluex_err: The flux err in parts per thousand.
        :param str: the outdir to store lk data
        """
        self.time = time
        self.flux = flux
        self.flux_err = flux_err
        self.outdir = outdir
        self.len = len(time)

    @classmethod
    def from_database(cls, tic: int, outdir: str):
        """Uses lightkurve to get TESS data for a TIC from MAST"""
        logger.info("Downloading LightCurveData from MAST")
        data = download_lightkurve_data(tic, outdir)
        logger.info("Completed light curve data download")
        data = data.stitch()
        data = data.remove_nans().remove_outliers(sigma=10)
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
            outdir=outdir,
        )

    @classmethod
    def from_cache(cls, tic: int, outdir: str):
        fname = LightCurveData.get_filepath(outdir)
        df = pd.read_csv(fname)
        logger.info(f"Lightcurve loaded from {fname}")
        return cls(
            time=np.array(df.time),
            flux=np.array(df.flux),
            flux_err=np.array(df.flux_err),
            outdir=outdir,
        )

    def to_dict(self):
        return dict(time=self.time, flux=self.flux, flux_err=self.flux_err)

    def save_data(self, outdir):
        fpath = self.get_filepath(outdir)
        df = pd.DataFrame(self.to_dict())
        df.to_csv(fpath, index=False)

    @staticmethod
    def get_filepath(outdir, fname="lightcurve.csv"):
        return os.path.join(outdir, fname)

    def get_observation_durations(self, tic):
        data = download_lightkurve_data(tic, self.outdir)
        observation_durations = []
        for obs in data:
            t = obs.time.value
            observation_durations.append([min(t), max(t)])
        return np.array(observation_durations)

    def _repr_html_(self):
        return (
            f"{self.__class__.__name__}: "
            f"{self.len} datapoints ({self.mem_size})"
        )


def search_for_lightkurve_data(tic):
    search = lk.search_lightcurve(
        target=f"TIC {tic}", mission="TESS", author="SPOC"
    )
    if not search:
        raise ValueError(f"Search contains no data products")
    logger.debug(f"Search  succeeded: {search}")

    # Restrict to short cadence no "fast" cadence
    search = search[np.where(search.table["t_exptime"] == 120)]
    if len(search) < 1:
        raise ValueError(
            f"Search contains no data products with t_exptime == 120"
        )

    return search


def download_lightkurve_data(tic, outdir):
    search = search_for_lightkurve_data(tic)
    logger.info(
        f"Downloading {len(search)} observations of light curve data "
        f"(TIC {tic})"
    )
    cache_dir = get_cache_dir(default=outdir)

    # see lightkurve docs on quality flags:
    # http://docs.lightkurve.org/reference/api/lightkurve.SearchResult.download_all.html
    data = search.download_all(
        download_dir=cache_dir,
        flux_column="pdcsap_flux",
        quality_bitmask="hardest",
    )
    if data is None:
        raise ValueError(f"No light curves for TIC {tic}")
    return data
