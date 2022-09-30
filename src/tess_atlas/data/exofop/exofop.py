import logging

from typing import List

import pandas as pd
from tess_atlas.utils import NOTEBOOK_LOGGER_NAME
from .tic_database import TICDatabase

from .paths import TIC_SEARCH
from .keys import TIC_ID, TOI, LK_AVAIL, SINGLE, MULTIPLANET

import functools

logger = logging.getLogger(NOTEBOOK_LOGGER_NAME)


@functools.lru_cache()
def get_tic_database(clean=False):
    return TICDatabase(clean=clean).df


def get_tic_id_for_toi(toi_number: int) -> int:
    tic_db = get_tic_database()
    toi = tic_db[tic_db[TOI] == toi_number + 0.01].iloc[0]
    return int(toi[TIC_ID])


@functools.lru_cache()
def get_toi_numbers_for_different_categories():
    tic_db = get_tic_database()
    tic_db = filter_db_without_lk(tic_db, remove=True)
    multi = tic_db[tic_db[MULTIPLANET]]
    single = tic_db[tic_db[SINGLE]]
    norm = tic_db[(~tic_db[SINGLE]) & (~tic_db[MULTIPLANET])]
    dfs = [multi, single, norm]
    toi_dfs = {}
    for df, name in zip(dfs, ["multi", "single", "norm"]):
        toi_ids = list(set(df[TOI].astype(int)))
        toi_dfs[name] = pd.DataFrame(dict(toi_numbers=toi_ids))
    return toi_dfs


def get_tic_data_from_database(toi_numbers: List[int]) -> pd.DataFrame:
    """Get rows of about a TIC  from ExoFOP associated with a TOI target.
    :param int toi_numbers: The list TOI number for which the TIC data is required
    :return: Dataframe with all TOIs for the TIC which contains TOI {toi_id}
    :rtype: pd.DataFrame
    """
    tic_db = get_tic_database()
    tics = [get_tic_id_for_toi(toi) for toi in toi_numbers]
    dfs = [tic_db[tic_db[TIC_ID] == tic].sort_values(TOI) for tic in tics]
    tois_for_tic = pd.concat(dfs)
    if len(tois_for_tic) < 1:
        raise ValueError(f"TOI data for TICs-{tics} does not exist.")
    return tois_for_tic


def get_tic_url(tic_id):
    """ExoFop Url"""
    return TIC_SEARCH.format(tic_id=tic_id)


def filter_db_without_lk(db, remove=True):
    if remove:
        db = db[db[LK_AVAIL] == True]
    return db


def get_toi_list(remove_toi_without_lk=True):
    db = get_tic_database()
    db = filter_db_without_lk(db, remove_toi_without_lk)
    return list(set(db[TOI].values.astype(int)))
