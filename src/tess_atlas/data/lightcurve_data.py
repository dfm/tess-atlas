import logging
import os

import lightkurve as lk
import numpy as np
import pandas as pd

from tess_atlas.utils import NOTEBOOK_LOGGER_NAME

logger = logging.getLogger(NOTEBOOK_LOGGER_NAME)


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
        logger.info(
            f"Searching for lightkurve data with target='TIC {tic}', "
            "mission='TESS'"
        )
        search = lk.search_lightcurve(target=f"TIC {tic}", mission="TESS")
        logger.debug(f"Search  succeeded: {search}")

        # Restrict to short cadence no "fast" cadence
        search = search[np.where(search.table["t_exptime"] == 120)]

        logger.info(
            f"Downloading {len(search)} observations of light curve data "
            f"(TIC {tic})"
        )
        data = search.download_all()
        if data is None:
            raise ValueError(f"No light curves for TIC {tic}")
        logger.info("Completed light curve data download")
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
        pd.DataFrame(self.to_dict()).to_csv(
            os.path.join(outdir, "lightcurve.csv"), index=False
        )
        logger.info(f"Saved lightcurve data.")
