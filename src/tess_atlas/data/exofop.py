import logging
import os
from typing import List

import pandas as pd
from tess_atlas.utils import NOTEBOOK_LOGGER_NAME
import lightkurve as lk


logger = logging.getLogger(NOTEBOOK_LOGGER_NAME)

EXOFOP = "https://exofop.ipac.caltech.edu/tess/"
TIC_DATASOURCE = EXOFOP + "download_toi.php?sort=toi&output=csv"
TIC_SEARCH = EXOFOP + "target.php?id={tic_id}"

DIR = os.path.dirname(__file__)


def get_tic_database(clean=False, check_lk=False):
    # if we have a cached database file
    cached_file = os.path.join(DIR, "cached_tic_database.csv")
    if os.path.isfile(cached_file) and not clean:
        db = pd.read_csv(cached_file)
    else:
        # go online to grab database and cache
        db = pd.read_csv(TIC_DATASOURCE)
        db.to_csv(cached_file, index=False)
    db[["TOI int", "planet count"]] = (
        db["TOI"].astype(str).str.split(".", 1, expand=True)
    )
    db = db.astype({"TOI int": "int", "planet count": "int"})
    db["Multiplanet System"] = db["TOI int"].duplicated(keep=False)
    db["Single Transit"] = db["Period (days)"] <= 0
    if check_lk:
        db["Lightcurve Availible"] = [is_lightcurve_availible_for_tic()]
    return db


def get_tic_id_for_toi(toi_number: int) -> int:
    tic_db = get_tic_database()
    toi = tic_db[tic_db["TOI"] == toi_number + 0.01].iloc[0]
    return int(toi["TIC ID"])


def get_toi_numbers_for_different_categories():
    tic_db = get_tic_database()
    multi = tic_db[tic_db["Multiplanet System"]]
    sing = tic_db[tic_db["Single Transit"]]
    basic = tic_db[
        (~tic_db["Single Transit"]) & (~tic_db["Multiplanet System"])
    ]
    dfs = [multi, sing, basic]
    toi_dfs = {}
    for df, name in zip(dfs, ["multi", "single", "basic"]):
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


def get_toi_list():
    database = get_tic_database()
    return list(set(database["TOI"].values.astype(int)))


def is_lightcurve_availible(tic):
    search = lk.search_lightcurve(
        target=f"TIC {tic}", mission="TESS", author="SPOC"
    )
    if len(search) > 1:
        return True
    return False
