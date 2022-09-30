import pandas as pd
import os

import logging
from tess_atlas.utils import NOTEBOOK_LOGGER_NAME, all_logging_disabled
from tess_atlas.data.data_utils import get_file_timestamp
from .paths import TIC_CACHE, TIC_DATASOURCE, TIC_OLD_CACHE
from .keys import (
    TIC_ID,
    LK_AVAIL,
    TOI,
    TOI_INT,
    PLANET_COUNT,
    MULTIPLANET,
    SINGLE,
    PERIOD,
)
from ..lightcurve_data.lightcurve_search import LightcurveSearch
from numpy import nan
from math import isnan
from tqdm.auto import tqdm
import random
import matplotlib.pyplot as plt

logger = logging.getLogger(NOTEBOOK_LOGGER_NAME)


class TICDatabase:
    """Interface to the table of TIC that we analyse"""

    def __init__(self, clean=False, update=False):
        self._db = pd.DataFrame()
        self.load(clean)
        if update:
            self.update()

    def load(self, clean=False):
        if not self.cache_present():
            clean = True  # if no cache, we must clean-load from exofop
        if clean:
            self._clean_cache()
            self.update()
        cache_time = get_file_timestamp(TIC_CACHE)
        logger.debug(f"Loading cached TIC list (last modified {cache_time})")
        self._db = pd.read_csv(TIC_CACHE)

    @staticmethod
    def cache_present():
        return os.path.isfile(TIC_CACHE)

    def _clean_cache(self):
        if self.cache_present():
            os.rename(TIC_CACHE, TIC_OLD_CACHE)

    def load_old_cache(self):
        if os.path.isfile(TIC_OLD_CACHE):
            return pd.read_csv(TIC_OLD_CACHE)
        return pd.DataFrame()

    @property
    def cached_tic_lk_dict(self):
        if self.cache_present():
            cached = pd.read_csv(TIC_CACHE)
            tics, lk_avail = cached[TIC_ID], cached[LK_AVAIL]
            return dict(zip(tics, lk_avail))
        return {}

    def update(self):
        table = _download_exofop_tic_table()
        tic_lk_dict = dict(zip(table[TIC_ID], table[LK_AVAIL]))
        tic_lk_dict.update(self.cached_tic_lk_dict)  # add any cached TICs
        self._clean_cache()  # after extracting cache data, we can clean cache

        # update the table with the lightcurve availability
        nan_lk_dict = {tic: lk for tic, lk in tic_lk_dict.items() if isnan(lk)}
        num_nan = len(nan_lk_dict)

        # shuffle dict
        nan_lk_dict = {
            k: nan_lk_dict[k]
            for k in random.sample(list(nan_lk_dict.keys()), num_nan)
        }
        for i, (t, avail) in tqdm(
            enumerate(nan_lk_dict.items()),
            total=num_nan,
            desc="Checking TIC lightcurve availability",
        ):
            if isnan(avail):
                nan_lk_dict[t] = _lightcurve_availible(t)
            if i % 100 == 0:  # update cache every 100 TICs
                logger.debug(
                    f"Updating TIC cache at {100 * (i / num_nan):.2f}%"
                )
                table[LK_AVAIL] = table[TIC_ID].map(nan_lk_dict)
                table.to_csv(TIC_CACHE, index=False)

        table[LK_AVAIL] = table[TIC_ID].map(tic_lk_dict)
        table.to_csv(TIC_CACHE, index=False)
        logger.info(
            f"Updated database from {len(self.load_old_cache())}-->{len(table)} TICs"
        )
        self.load()

    @property
    def df(self):
        return self._db

    def plot_caches(self):
        old = self.load_old_cache()
        new = self._db
        if len(old) > 0:
            fig, axes = plt.subplots(
                2, 1, figsize=(10, 5), sharex=True, sharey=True
            )
            _plot_lk_status(axes[1], old, "Old Cache")
            ax = axes[0]
        else:
            fig, axes = plt.subplots(1, 1, figsize=(10, 2.5))
            ax = axes
        _plot_lk_status(ax, new, label="Cache")
        plt.suptitle("TOIs with 2-min Lightcurve data", fontsize="xx-large")
        plt.tight_layout()
        plt.savefig(TIC_CACHE.replace(".csv", ".png"))


def _download_exofop_tic_table():
    db = pd.read_csv(TIC_DATASOURCE)
    logger.info(f"TIC database has {len(db)} entries")
    db[[TOI_INT, PLANET_COUNT]] = (
        db[TOI].astype(str).str.split(".", 1, expand=True)
    )
    db = db.astype({TOI_INT: "int", PLANET_COUNT: "int"})
    db[MULTIPLANET] = db[TOI_INT].duplicated(keep=False)
    db[SINGLE] = db[PERIOD] <= 0
    db[LK_AVAIL] = nan
    return db


def _lightcurve_availible(tic):
    with all_logging_disabled():
        try:
            return LightcurveSearch(tic).lk_data_available
        except Exception as _:
            return nan


def _plot_lk_status(ax, data, label=""):
    r = dict(ymin=0, ymax=2, lw=0.1, alpha=0.5)
    valid = set(data[data[LK_AVAIL] == True][TOI_INT].tolist())
    invalid = set(data[data[LK_AVAIL] == False][TOI_INT].tolist())
    total = len(data)
    ax.vlines(
        list(valid),
        **r,
        label=f"Valid ({len(valid)}/{total} TOIs)",
        color="tab:green",
    )
    ax.vlines(
        list(invalid),
        **r,
        label=f"Invalid ({len(invalid)}/{total} TOIs)",
        color="tab:red",
        zorder=-10,
    )
    ax.legend(
        loc="upper left",
    )
    ax.set_ylim(0.99, 1.01)
    ax.set_xlim(left=100, right=max(data["TOI int"]))
    ax.set_yticks([])
    ax.set_xlabel("TOI Number")
    ax.set_ylabel(label)
