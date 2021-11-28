import logging
import os
from typing import List, Optional

import pandas as pd
from arviz import InferenceData
from IPython.display import HTML, display

from tess_atlas.utils import NOTEBOOK_LOGGER_NAME

from .data_object import DataObject
from .exofop import get_tic_data_from_database, get_tic_url
from .inference_data_tools import (
    get_idata_fname,
    load_inference_data,
    save_inference_data,
)
from .lightcurve_data import LightCurveData
from .planet_candidate import PlanetCandidate
from .stellar_data import StellarData

logger = logging.getLogger(NOTEBOOK_LOGGER_NAME)

TOI_DIR = "toi_{toi}_files"


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
        if os.path.isfile(get_idata_fname(self.outdir)):
            self.inference_data = load_inference_data(self.outdir)
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
        return get_tic_url(self.tic_number)

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

    def save_data(self, inference_data: Optional[InferenceData] = None):
        fpath = self.get_filepath(self.outdir)
        self.tic_data.to_csv(fpath)
        self.lightcurve.save_data(self.outdir)
        self.stellar_data.save_data(self.outdir)
        if inference_data is not None:
            self.inference_data = inference_data
            save_inference_data(inference_data, self.outdir)
        logger.info(f"Saved data in {self.outdir}")
