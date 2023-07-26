import logging
import os
import random
from math import isnan
from typing import Dict, List, NamedTuple, Optional, Union

import pandas as pd
from numpy import nan
from tqdm.auto import tqdm

from tess_atlas.data.exofop.constants import (
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

from ...file_management import get_file_timestamp
from ...logger import LOGGER_NAME, all_logging_disabled
from ..lightcurve_data.lightcurve_search import LightcurveSearch
from .plotting import plot_lk_status

logger = logging.getLogger(LOGGER_NAME)


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

    def __init__(self, clean=False, update=False, fname=TIC_CACHE):
        self._db = pd.DataFrame()
        self.fname = fname
        self.previous_fname = fname + ".old"
        self.load(clean)
        if update:
            self.update()

    def load(self, clean=False):
        if not self.cache_present:
            clean = True  # if no cache, we must clean-load from exofop
        if clean:
            self._clean_cache()
            self.update()
        logger.debug(
            f"Loading cached TIC list (last modified {self.cache_timestamp})"
        )
        self._db = pd.read_csv(self.fname)
        return self

    @property
    def cache_timestamp(self):
        return get_file_timestamp(self.fname)

    @property
    def cache_present(self):
        return os.path.isfile(self.fname)

    def _clean_cache(self):
        if self.cache_present:
            os.rename(self.fname, self.previous_fname)

    def load_old_cache(self):
        if os.path.isfile(self.previous_fname):
            return pd.read_csv(self.previous_fname)
        return pd.DataFrame()

    @property
    def cached_tic_lk_dict(self):
        if self.cache_present:
            cached = pd.read_csv(self.fname)
            tics, lk_avail = cached[TIC_ID], cached[LK_AVAIL]
            return dict(zip(tics, lk_avail))
        return {}

    def update(self, save_name=TIC_CACHE):
        """Update the TIC cache from the exofop database

        Queries Lightkurve to check if lightcurve data is available for each TIC.
        (querying lightkurve is slow, so we cache the results)
        """
        new_table = _download_exofop_tic_table()
        logger.info(
            f"Current TIC cache: {len(self._db):,}, online TIC list: {len(new_table):,}"
        )

        # Update new tables' TIC:lk-avail dict with cached data (for TIC already checked)
        tic_lk_dict = dict(zip(new_table[TIC_ID], new_table[LK_AVAIL]))
        tic_lk_dict.update(self.cached_tic_lk_dict)

        self._clean_cache()  # after extracting cache data, we can clean cache
        self._save_table_with_lk_status(new_table, tic_lk_dict, save_name)

        # Get the subset of TICs with no lightcurve data
        nan_lk_dict = {tic: lk for tic, lk in tic_lk_dict.items() if isnan(lk)}
        num_nan = len(nan_lk_dict)
        logger.info(
            f"Checking {num_nan}/{len(new_table)} TICs for lightcurve availability"
        )
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
                nan_lk_dict[t] = _lightcurve_available(t)
            if i % 100 == 0:  # update cache every 100 TICs
                logger.debug(
                    f"Updating TIC cache at {100 * (i / num_nan):.2f}%"
                )
                self._save_table_with_lk_status(
                    new_table, nan_lk_dict, save_name
                )

        self._save_table_with_lk_status(new_table, nan_lk_dict, save_name)
        logger.info(
            f"Updated database from {len(self.load_old_cache())}-->{len(new_table)} TICs"
        )
        self.fname = save_name
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
        toi_data = self._db[self._db[TOI] == toi_number + 0.01]
        if len(toi_data) == 0:
            raise ValueError(f"TOI {toi_number} not found in database")
        return int(toi_data.iloc[0][TIC_ID])

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

    def plot(self, new=None, old=None) -> "plt.Figure":
        """Plot if the TOI has lightcurve data using the old and new TIC caches"""
        if new is None:
            new = self._db
        if old is None:
            old = self.load_old_cache()
        return plot_lk_status(new, old)

    def get_counts(self, filter=False) -> Categories:
        """Get the number of TOIs in the database"""
        return Categories(
            multiplanet=len(self.get_df(MULTIPLANET, filter)),
            single_transit=len(self.get_df(SINGLE_TRANSIT, filter)),
            normal=len(self.get_df(NORMAL, filter)),
            all=len(self.get_df(None, filter)),
        )

    def _save_table_with_lk_status(
        self,
        table: pd.DataFrame,
        lk_status_dict: Dict[int, bool],
        save_fn=TIC_CACHE,
    ):
        """Update the table with the lightcurve status"""
        tic_lk_dict = dict(zip(table[TIC_ID], table[LK_AVAIL]))
        tic_lk_dict.update(lk_status_dict)
        table[LK_AVAIL] = table[TIC_ID].map(tic_lk_dict)
        table.to_csv(save_fn, index=False)
        self.plot(table, self.load_old_cache())
        return table

    @property
    def n_tois(self):
        return len(self.get_toi_list(remove_toi_without_lk=True))

    def __repr__(self):
        return f"<Exofop Table ({self.n_tois} TOIs)>"


def _download_exofop_tic_table() -> pd.DataFrame:
    db = pd.read_csv(TIC_DATASOURCE)
    logger.info(f"TIC database has {len(db)} entries")
    db[[TOI_INT, PLANET_COUNT]] = (
        db[TOI].astype(str).str.split(".", n=1, expand=True).astype(int)
    )
    db = db.astype({TOI_INT: "int", PLANET_COUNT: "int"})
    db[MULTIPLANET] = db[TOI_INT].duplicated(keep=False)
    db[SINGLE_TRANSIT] = db[PERIOD] <= 0
    db[LK_AVAIL] = nan
    return db


def _lightcurve_available(tic: int) -> Union[bool, float]:
    """Check if a TIC has lightcurve data available

    Returns:
        Union[bool, float]: True if lightcurve data is available, False if not, nan if error
    """
    with all_logging_disabled():
        try:
            return LightcurveSearch(tic).lk_data_available
        except Exception as _:
            return nan


def _filter_db_without_lk(db: pd.DataFrame) -> pd.DataFrame:
    """Filter out TOIs without lightcurve data"""
    return db[db[LK_AVAIL] == True]
