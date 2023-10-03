import logging
import os
from typing import List, NamedTuple, Optional, Union

import pandas as pd

from tess_atlas.data.exofop.constants import (
    LK_AVAIL,
    MULTIPLANET,
    NORMAL,
    SINGLE_TRANSIT,
    TIC_CACHE,
    TIC_CACHE_URL,
    TIC_ID,
    TIC_SEARCH,
    TOI,
)

from ...file_management import get_file_timestamp
from ...logger import LOGGER_NAME, all_logging_disabled
from ..lightcurve_data.lightcurve_search import LightcurveSearch

logger = logging.getLogger(LOGGER_NAME)


class Categories(NamedTuple):
    multiplanet: Union[int, List[int]]
    single_transit: Union[int, List[int]]
    normal: Union[int, List[int]]
    all: Optional[Union[int, List[int]]] = None


class ExofopDatabase:
    """Interface to the Exofop datatable of TIC that we analyse."""

    def __init__(self, clean=False, cache_fname=TIC_CACHE):
        self._db = pd.DataFrame()
        self.cache_fname = cache_fname
        self.load(clean)

    def load(self, clean=False):
        if not self.cache_present or clean:
            self._clean_cache()
            self.__download()
        logger.debug(
            f"Loading cached TIC list (last modified {self.cache_timestamp})"
        )
        self._db = pd.read_csv(self.cache_fname)
        return self

    @property
    def cache_timestamp(self):
        return get_file_timestamp(self.cache_fname)

    @property
    def cache_present(self):
        return os.path.isfile(self.cache_fname)

    def _clean_cache(self):
        if self.cache_present:
            os.remove(self.cache_fname)

    @property
    def cached_tic_lk_dict(self):
        if self.cache_present:
            cached = pd.read_csv(self.cache_fname)
            tics, lk_avail = cached[TIC_ID], cached[LK_AVAIL]
            return dict(zip(tics, lk_avail))
        return {}

    def __download(self):
        """Download the TIC cache.

        Note: this contains the LK_AVAIL column,
        unlike the original ExoFOP table
        """
        df = pd.read_csv(TIC_CACHE_URL)
        df.to_csv(self.cache_fname, index=False)

    def get_df(
        self, category=None, remove_toi_without_lk=False
    ) -> pd.DataFrame:
        df = self._db.copy()
        if remove_toi_without_lk:
            df = df[df[LK_AVAIL] == True]
        if category is not None:
            if category == NORMAL:
                df = df[~df[MULTIPLANET] & ~df[SINGLE_TRANSIT]]
            else:
                df = df[df[category]]
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

    def get_counts(self, filter=False) -> Categories:
        """Get the number of TOIs in the database"""
        return Categories(
            multiplanet=len(self.get_df(MULTIPLANET, filter)),
            single_transit=len(self.get_df(SINGLE_TRANSIT, filter)),
            normal=len(self.get_df(NORMAL, filter)),
            all=len(self.get_df(None, filter)),
        )

    @property
    def n_tois(self):
        return len(self.get_toi_list(remove_toi_without_lk=True))

    def __repr__(self):
        return f"<Exofop Table ({self.n_tois} TOIs)>"
