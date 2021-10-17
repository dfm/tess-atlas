import logging
import os
from typing import Dict, List, Optional

import pandas as pd
from IPython.display import display
from IPython.display import HTML

from pymc3.sampling import MultiTrace


from .data_object import DataObject

from tess_atlas.utils import NOTEBOOK_LOGGER_NAME
from .lightcurve_data import LightCurveData
from .planet_candidate import PlanetCandidate
from .stellar_data import StellarData

from .inference_data import InferenceData


logger = logging.getLogger(NOTEBOOK_LOGGER_NAME)

EXOFOP = "https://exofop.ipac.caltech.edu/tess/"
TIC_DATASOURCE = EXOFOP + "download_toi.php?sort=toi&output=csv"
TIC_SEARCH = EXOFOP + "target.php?id={tic_id}"

TOI_DIR = "toi_{toi}_files"

DIR = os.path.dirname(__file__)


def get_tic_database():
    # if we have a cached database file
    cached_file = os.path.join(DIR, "cached_tic_database.csv")
    if os.path.isfile(cached_file):
        db = pd.read_csv(cached_file)
    else:
        # go online to grab database and cache
        db = pd.read_csv(TIC_DATASOURCE)
        db.to_csv(cached_file)
    return db


def get_tic_id_for_toi(toi_number: int) -> int:
    tic_db = get_tic_database()
    toi = tic_db[tic_db["TOI"] == toi_number + 0.01].iloc[0]
    return int(toi["TIC ID"])


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


class TICEntry(DataObject):
    """Hold information about a TIC (TESS Input Catalog) entry"""

    def __init__(
        self,
        toi: int,
        tic_data: pd.DataFrame,
        loaded_from_cache: Optional[bool] = False,
    ):
        self.toi_number = toi
        self.tic_data = tic_data
        self.outdir = TOI_DIR.format(toi=toi)
        data_kwargs = dict(tic=self.tic_number, outdir=self.outdir)
        self.lightcurve = LightCurveData.load(**data_kwargs)
        self.stellar_data = StellarData.load(**data_kwargs)
        self.inference_data = None
        if os.path.isfile(InferenceData.get_filepath(self.outdir)):
            self.inference_data = InferenceData.load(self.outdir)
        self.candidates = self.get_candidates()
        self.loaded_from_cache = loaded_from_cache
        if not self.loaded_from_cache:
            self.save_data()

    def get_candidates(self) -> List[PlanetCandidate]:
        candidates = []
        for index, toi_data in self.tic_data.iterrows():
            candidate = PlanetCandidate.from_database(
                toi_data=toi_data.to_dict(), lightcurve=self.lightcurve
            )
            candidates.append(candidate)
        return candidates

    @property
    def exofop_url(self) -> str:
        return TIC_SEARCH.format(tic_id=self.tic_number)

    @property
    def tic_number(self) -> int:
        try:
            return int(self.tic_data["TIC ID"].iloc[0])
        except Exception:
            print(self.tic_data)

    @property
    def planet_count(self) -> int:
        return len(self.candidates)

    @classmethod
    def load(cls, toi: int):
        toi_dir = TOI_DIR.format(toi=toi)
        cache_fn = TICEntry.get_filepath(toi_dir)
        if TICEntry.cached_data_present(cache_fn):
            return cls.from_cache(toi, toi_dir)
        else:
            return cls.from_database(toi)

    @classmethod
    def from_database(cls, toi: int):
        logger.info("Querying ExoFOP for TIC data")
        return cls(toi=toi, tic_data=get_tic_data_from_database([toi]))

    @classmethod
    def from_cache(cls, toi: int, outdir: str):
        fpath = TICEntry.get_filepath(outdir)
        logger.info("Loading cached data")
        return cls(
            toi=toi, tic_data=pd.read_csv(fpath), loaded_from_cache=True
        )

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [candidate.to_dict() for candidate in self.candidates]
        )

    def display(self):
        df = self.to_dataframe()
        df = df.transpose()
        df.columns = df.loc["TOI"]
        display(df)
        more_data_str = "More data on ExoFOP page"
        html_str = f"<a href='{self.exofop_url}'>{more_data_str}</a>"
        display(HTML(html_str))

    @property
    def outdir(self):
        return self.__outdir

    @outdir.setter
    def outdir(self, outdir):
        os.makedirs(outdir, exist_ok=True)
        self.__outdir = outdir

    @staticmethod
    def get_filepath(outdir, fname="tic_data.csv"):
        return os.path.join(outdir, fname)

    def save_data(self, trace=None):
        fpath = self.get_filepath(self.outdir)
        self.tic_data.to_csv(fpath)
        self.lightcurve.save_data(self.outdir)
        self.stellar_data.save_data(self.outdir)
        if trace is not None:
            self.inference_data = InferenceData(trace)
            self.inference_data.save_data(self.outdir)
