import logging

import lightkurve as lk
import numpy as np

from ...logger import LOGGER_NAME
from ...utils import get_cache_dir

logger = logging.getLogger(LOGGER_NAME)


class LightcurveSearch:
    def __init__(self, tic: int):
        self.tic = tic
        self.search = lk_search(tic)

    @property
    def lk_data_available(self) -> bool:
        return len(self.search) > 0

    def lk_download(self, outdir: str) -> lk.LightCurve:
        if not self.lk_data_available:
            raise ValueError(
                f"No lightkurve data available for TIC {self.tic}"
            )

        # note: doesn't matter which planet's TIC we use for multi-planet system
        logger.info(
            f"Downloading {len(self.search)} observations of light curve data "
            f"(TIC {self.tic})"
        )
        cache_dir = get_cache_dir(default=outdir)

        # see lightkurve docs on quality flags:
        # http://docs.lightkurve.org/reference/api/lightkurve.SearchResult.download_all.html
        data = self.search.download_all(
            download_dir=cache_dir,
            flux_column="pdcsap_flux",
            quality_bitmask="hardest",
        )
        if data is None:
            raise ValueError(f"No light curves for TIC {self.tic}")
        data = data.stitch().remove_nans()
        return data


def lk_search(tic) -> lk.search.SearchResult:
    search = lk.search_lightcurve(
        target=f"TIC {tic}", mission="TESS", author="SPOC"
    )
    if not search:
        logger.error(f"Search contains no data products")
    else:
        logger.debug(f"Search  succeeded: {search}")

        # Restrict to short cadence no "fast" cadence
        search = search[np.where(search.table["t_exptime"] == 120)]
        if len(search) < 1:
            logger.error(
                f"Search contains no data products with t_exptime == 120"
            )
    return search
