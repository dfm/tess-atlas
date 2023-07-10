import logging
import os
import random
from collections import namedtuple
from dataclasses import dataclass
from math import isnan
from typing import Any, Dict, List, NamedTuple, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
from numpy import nan
from tqdm.auto import tqdm

from tess_atlas.data.data_utils import get_file_timestamp
from tess_atlas.utils import NOTEBOOK_LOGGER_NAME, all_logging_disabled

from ..lightcurve_data.lightcurve_search import LightcurveSearch
from .constants import (
    LK_AVAIL,
    MULTIPLANET,
    NORMAL,
    PERIOD,
    PLANET_COUNT,
    SINGLE_TRANSIT,
    TIC_CACHE,
    TIC_DATASOURCE,
    TIC_ID,
    TIC_OLD_CACHE,
    TIC_SEARCH,
    TOI,
    TOI_INT,
)

logger = logging.getLogger(NOTEBOOK_LOGGER_NAME)


class Categories(NamedTuple):
    multiplanet: Union[int, List[int]]
    single_transit: Union[int, List[int]]
    normal: Union[int, List[int]]
    all: Optional[Union[int, List[int]]] = None


class ExofopDatabase:
    """Interface to the Exofop datatable of TIC that we analyse.

    We cache the TIC database in a CSV file, and update it periodically when needed.
    # TODO: cron job to update the cache periodically?

    """

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
        """Update the TIC cache from the exofop database

        Queries Lightkurve to check if lightcurve data is available for each TIC.
        (quering lightkurve is slow, so we cache the results)
        """
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

    def get_df(
        self, category=None, remove_toi_without_lk=False
    ) -> pd.DataFrame:
        df = self._db.copy()
        if remove_toi_without_lk:
            df = _filter_db_without_lk(df)
        if category is not None:
            if category == MULTIPLANET:
                df = df[df[MULTIPLANET]]
            elif category == SINGLE_TRANSIT:
                df = df[df[SINGLE_TRANSIT]]
            elif category == NORMAL:
                df = df[~df[MULTIPLANET] & ~df[SINGLE_TRANSIT]]
        return df

    def get_tic_id_for_toi(self, toi_number: int) -> int:
        """Get the TIC ID for a given TOI number"""
        toi = self._db[self._db[TOI] == toi_number + 0.01].iloc[0]
        return int(toi[TIC_ID])

    def get_categorised_toi_lists(self) -> Categories:
        """Get the TOI numbers for different categories of TICs.

        Returns:
            Categories:
                A Categories object with each category storing its "toi_numbers"
        """
        return Categories(
            multiplanet=self.get_toi_list(category=MULTIPLANET),
            single_transit=self.get_toi_list(category=SINGLE_TRANSIT),
            normal=self.get_toi_list(category=NORMAL),
            all=self.get_toi_list(category=None),
        )

    def get_toi_list(
        self, category=None, remove_toi_without_lk=True
    ) -> List[int]:
        """Get the list of TOI numbers in the database"""
        db = self.get_df(
            category=category, remove_toi_without_lk=remove_toi_without_lk
        )
        return list(set(db[TOI].values.astype(int)))

    def get_tic_data(self, toi_numbers: List[int]) -> pd.DataFrame:
        """Get rows of about a TIC  from ExoFOP associated with a TOI target.

        :param int toi_numbers: The list TOI number for which the TIC data is required
        :return: Dataframe with all TOIs for the TIC which contains TOI {toi_id}
        :rtype: pd.DataFrame
        """
        tic_db = self._db
        tics = [self.get_tic_id_for_toi(toi) for toi in toi_numbers]
        dfs = [tic_db[tic_db[TIC_ID] == tic].sort_values(TOI) for tic in tics]
        tois_for_tic = pd.concat(dfs)
        if len(tois_for_tic) < 1:
            raise ValueError(f"TOI data for TICs-{tics} does not exist.")
        return tois_for_tic

    @staticmethod
    def get_tic_url(tic_id):
        """ExoFop Url"""
        return TIC_SEARCH.format(tic_id=tic_id)

    def plot_caches(self) -> None:
        """Plot if the TOI has lightcurve data using the old and new TIC caches"""
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

    def get_counts(self, filter=False) -> Categories:
        """Get the number of TOIs in the database"""
        return Categories(
            multiplanet=len(self.get_df(MULTIPLANET, filter)),
            single_transit=len(self.get_df(SINGLE_TRANSIT, filter)),
            normal=len(self.get_df(NORMAL, filter)),
            all=len(self.get_df(None, filter)),
        )


def _download_exofop_tic_table() -> pd.DataFrame:
    db = pd.read_csv(TIC_DATASOURCE)
    logger.info(f"TIC database has {len(db)} entries")
    db[[TOI_INT, PLANET_COUNT]] = (
        db[TOI].astype(str).str.split(".", 1, expand=True)
    )
    db = db.astype({TOI_INT: "int", PLANET_COUNT: "int"})
    db[MULTIPLANET] = db[TOI_INT].duplicated(keep=False)
    db[SINGLE_TRANSIT] = db[PERIOD] <= 0
    db[LK_AVAIL] = nan
    return db


def _lightcurve_availible(tic: int) -> Union[bool, float]:
    """Check if a TIC has lightcurve data available

    Returns:
        Union[bool, float]: True if lightcurve data is available, False if not, nan if error
    """
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


def _filter_db_without_lk(db: pd.DataFrame) -> pd.DataFrame:
    """Filter out TOIs without lightcurve data"""
    return db[db[LK_AVAIL] == True]
