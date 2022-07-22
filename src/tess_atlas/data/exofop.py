import logging
import os
from typing import List
from math import nan
from collections import OrderedDict

import pandas as pd
from tess_atlas.utils import NOTEBOOK_LOGGER_NAME
from tess_atlas.data.data_utils import get_file_timestamp
import lightkurve as lk
import functools
from tqdm.auto import tqdm

logger = logging.getLogger(NOTEBOOK_LOGGER_NAME)

EXOFOP = "https://exofop.ipac.caltech.edu/tess/"
TIC_DATASOURCE = EXOFOP + "download_toi.php?sort=toi&output=csv"
TIC_SEARCH = EXOFOP + "target.php?id={tic_id}"

DIR = os.path.dirname(__file__)
TIC_CACHE = os.path.join(DIR, "cached_tic_database.csv")


@functools.lru_cache()
def get_tic_database(clean=False):
    """
    TIC data with columns:
        'TIC ID'
        'TOI' (there can be duplicates )
        'Previous CTOI'
        'Master'
        'SG1A'
        'SG1B'
        'SG2'
        'SG3'
        'SG4'
        'SG5'
        'ACWG ESM'
        'ACWG TSM'
        'Time Series Observations'
        'Spectroscopy Observations'
        'Imaging Observations'
        'TESS Disposition'
        'TFOPWG Disposition'
        'TESS Mag'
        'TESS Mag err'
        'Planet Name'
        'Pipeline Signal ID'
        'Source'
        'Detection'
        'RA'
        'Dec'
        'PM RA (mas/yr)'
        'PM RA err (mas/yr)'
        'PM Dec (mas/yr)'
        'PM Dec err (mas/yr)'
        'Epoch (BJD)'
        'Epoch (BJD) err'
        'Period (days)'
        'Period (days) err'
        'Duration (hours)'
        'Duration (hours) err'
        'Depth (mmag)'
        'Depth (mmag) err'
        'Depth (ppm)'
        'Depth (ppm) err'
        'Planet Radius (R_Earth)'
        'Planet Radius (R_Earth) err'
        'Planet Insolation (Earth Flux)'
        'Planet Equil Temp (K)'
        'Planet SNR'
        'Stellar Distance (pc)'
        'Stellar Distance (pc) err'
        'Stellar Eff Temp (K)'
        'Stellar Eff Temp (K) err'
        'Stellar log(g) (cm/s^2)'
        'Stellar log(g) (cm/s^2) err'
        'Stellar Radius (R_Sun)'
        'Stellar Radius (R_Sun) err'
        'Stellar Metallicity'
        'Stellar Metallicity err'
        'Stellar Mass (M_Sun)'
        'Stellar Mass (M_Sun) err'
        'Sectors'
        'Date TOI Alerted (UTC)'
        'Date TOI Updated (UTC)'
        'Date Modified'
        'Comments'
        'TOI int'
        'planet count'
        'Multiplanet System'
        'Single Transit'
        'Lightcurve Availible'
    """
    if clean and tic_cache_present():
        os.remove(TIC_CACHE)

    # if we have a cached database file
    if tic_cache_present():
        cache_time = get_file_timestamp(TIC_CACHE)
        logger.debug(f"Loading cached TIC list (last modified {cache_time})")
        db = pd.read_csv(TIC_CACHE)
    else:
        db = update_local_tic_database()

    return db


def get_tic_id_for_toi(toi_number: int) -> int:
    tic_db = get_tic_database()
    toi = tic_db[tic_db["TOI"] == toi_number + 0.01].iloc[0]
    return int(toi["TIC ID"])


@functools.lru_cache()
def get_toi_numbers_for_different_categories():
    tic_db = get_tic_database()
    tic_db = filter_db_without_lk(tic_db, remove=True)
    multi = tic_db[tic_db["Multiplanet System"]]
    single = tic_db[tic_db["Single Transit"]]
    norm = tic_db[
        (~tic_db["Single Transit"]) & (~tic_db["Multiplanet System"])
    ]
    dfs = [multi, single, norm]
    toi_dfs = {}
    for df, name in zip(dfs, ["multi", "single", "norm"]):
        toi_ids = list(set(df["TOI"].astype(int)))
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
    dfs = [tic_db[tic_db["TIC ID"] == tic].sort_values("TOI") for tic in tics]
    tois_for_tic = pd.concat(dfs)
    if len(tois_for_tic) < 1:
        raise ValueError(f"TOI data for TICs-{tics} does not exist.")
    return tois_for_tic


def get_tic_url(tic_id):
    return TIC_SEARCH.format(tic_id=tic_id)


def filter_db_without_lk(db, remove=True):
    if remove:
        db = db[db["Lightcurve Availible"] == True]
    return db


def get_toi_list(remove_toi_without_lk=True):
    db = get_tic_database()
    db = filter_db_without_lk(db, remove_toi_without_lk)
    return list(set(db["TOI"].values.astype(int)))


def is_lightcurve_availible(tic):
    # TODO: would be better if we use
    # tess_atlas.lightcurve_data.search_for_lightkurve_data
    # however -- getting a silly import error -- probably recursive :(
    search = lk.search_lightcurve(
        target=f"TIC {tic}",
        mission="TESS",
        author="SPOC",
    )
    if len(search) > 1:
        return True
    return False


def tic_cache_present():
    return os.path.isfile(TIC_CACHE)


def get_db_lk_status(db):
    # unpack data
    data = db[["TIC ID", "Lightcurve Availible"]].to_dict("list")
    tic, lk_status = data["TIC ID"], data["Lightcurve Availible"]
    lk_status = {t: s for t, s in zip(tic, lk_status)}
    nan_list = [
        t for t, status in zip(tic, lk_status) if not isinstance(status, bool)
    ]

    print(f"Updating lk status for {len(nan_list)} TOIs")
    for t in tqdm(nan_list, desc="Checking TIC for lightcurve data"):
        lk_status[t] = is_lightcurve_availible(t)
    lk_status = [lk_status[t] for t in tic]
    return lk_status


def update_local_tic_database():
    # go online to grab database and cache
    db = pd.read_csv(TIC_DATASOURCE)
    print(f"TIC database has {len(db)} entries")
    db[["TOI int", "planet count"]] = (
        db["TOI"].astype(str).str.split(".", 1, expand=True)
    )
    db = db.astype({"TOI int": "int", "planet count": "int"})
    db["Multiplanet System"] = db["TOI int"].duplicated(keep=False)
    db["Single Transit"] = db["Period (days)"] <= 0

    if tic_cache_present():
        print("Updating TIC db")
        # then we add in the cached col of 'Lightcurve Availible'
        cache = pd.read_csv(TIC_CACHE)
        cache_len = len(cache)
        cache = cache[["TOI", "Lightcurve Availible"]]
        new_db = db.merge(cache, how="outer", on="TOI")
        new_db = new_db.drop_duplicates(subset="TOI", keep="last")
        new_db = new_db[new_db["TOI"].isin(db["TOI"].values.tolist())]
        if len(new_db) != len(db):
            raise ValueError(
                f"len(new) {len(new_db)} != len(expected) {len(db)}"
            )
        db = new_db.copy()
    else:
        db["Lightcurve Availible"] = nan
        cache_len = 0
    db["Lightcurve Availible"] = get_db_lk_status(db)
    db.to_csv(TIC_CACHE, index=False)
    assert db["Lightcurve Availible"].isnull().sum() == 0
    print(f"Updated database from {cache_len}-->{len(db)} TOIs")
    return db


if __name__ == "__main__":
    update_local_tic_database()
    print(f"Number of TOIs: {len(get_toi_list())}")
