import logging
import os
import shutil
from typing import Dict, List, Optional

import pandas as pd
from arviz import InferenceData

from ..file_management import TIC_CSV, TOI_DIR
from ..logger import LOGGER_NAME
from .data_object import DataObject
from .exofop import EXOFOP_DATA
from .inference_data_tools import (
    get_idata_fname,
    load_inference_data,
    save_inference_data,
)
from .lightcurve_data import LightCurveData
from .optimized_params import OptimizedParams
from .planet_candidate import PlanetCandidate
from .stellar_data import StellarData

logger = logging.getLogger(LOGGER_NAME)

TIC_ID = "TIC ID"


class TICEntry(DataObject):
    """Holds information about a TIC (TESS Input Catalog) entry"""

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
        self.optimized_params = OptimizedParams.load(**data_kwargs)
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
        return EXOFOP_DATA.get_tic_url(self.tic_number)

    @property
    def tic_number(self) -> int:
        try:
            return _get_tic_id_from_table(self.tic_data)
        except Exception as e:
            raise ValueError(
                f"Error {e}. "
                f"Could not get TIC number for TOI {self.toi_number}. "
                f"TIC data: \n{self.tic_data}"
            )

    @property
    def planet_count(self) -> int:
        return len(self.candidates)

    @classmethod
    def load(cls, toi: int, clean: Optional[bool] = False):
        toi_dir = TOI_DIR.format(toi=toi)
        cache_fn = TICEntry.get_filepath(toi_dir)

        if clean and os.path.isdir(toi_dir):
            shutil.rmtree(toi_dir)

        if TICEntry.cached_data_present(cache_fn):
            try:
                return cls.from_cache(toi, toi_dir)
            except Exception as e:
                logger.info(f"Error loading fom cache: ''{e}''")

        return cls.from_database(toi)

    @classmethod
    def from_database(cls, toi: int):
        logger.info("Querying ExoFOP for TIC data")
        tic_data = EXOFOP_DATA.get_tic_data([toi])
        return cls(toi=toi, tic_data=tic_data)

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

    def _repr_html_(self):
        df = self.to_dataframe()
        df = df.transpose()
        df.columns = df.loc["TOI"]
        return (
            f"{df._repr_html_()}<br>\n"
            # f"Memory: {self.mem_size}<br>\n"
            f"<a href='{self.exofop_url}'>More data on ExoFOP page</a>"
        )

    @property
    def outdir(self):
        return self.__outdir

    @outdir.setter
    def outdir(self, outdir):
        os.makedirs(outdir, exist_ok=True)
        self.__outdir = outdir

    @staticmethod
    def get_filepath(outdir, fname=TIC_CSV):
        return os.path.join(outdir, fname)

    def save_data(
        self,
        inference_data: Optional[InferenceData] = None,
        optimized_params: Optional[Dict] = None,
    ):
        fpath = self.get_filepath(self.outdir)
        self.tic_data.to_csv(fpath)
        self.lightcurve.save_data(self.outdir)
        self.stellar_data.save_data(self.outdir)
        if inference_data is not None:
            self.inference_data = inference_data
            save_inference_data(inference_data, self.outdir)
        if optimized_params is not None:
            self.optimized_params = OptimizedParams(
                optimized_params, self.outdir
            )
            self.optimized_params.save_data(self.outdir)
        logger.info(f"Saved data in {self.outdir}")


def _get_tic_id_from_table(tic_data: pd.DataFrame) -> int:
    return int(tic_data[TIC_ID].iloc[0])
