import logging
import os
from copy import deepcopy
from typing import Dict, List

import pandas as pd

from ..logger import LOGGER_NAME
from .data_object import DataObject
from .data_utils import load_json, save_json

logger = logging.getLogger(LOGGER_NAME)

OPTIMIZED_PARAMS_JSON = "optimized_params.json"


class OptimizedParams(DataObject):
    def __init__(self, params: Dict[str, List], outdir: str):
        self.params = params
        self.outdir = outdir

    @classmethod
    def from_database(cls, tic: int, outdir: str):
        return None  # nothing optimized yet!

    @classmethod
    def from_cache(cls, tic: int, outdir: str):
        fpath = OptimizedParams.get_filepath(outdir)
        data = load_json(fpath)
        logger.info(f"Optimized Params loaded from {fpath}")
        return cls(params=data, outdir=outdir)

    def save_data(self, outdir):
        save_json(self.get_filepath(outdir), self.to_dict())

    def __str__(self):
        return self.to_dict().__str__()

    def to_dict(self, remove_extras=False):
        d = deepcopy(self.params)
        if remove_extras:
            d = {k: v for k, v in d.items() if "__" not in k}
        return d

    def to_dataframe(self):
        d = self.to_dict(remove_extras=True)
        u = d.pop("u")
        val_len = max([len(v) for v in d.values() if isinstance(v, list)])
        for k, v in d.items():
            if not isinstance(v, list):
                d[k] = [v] * val_len
        d["u"] = [u] * val_len
        df = pd.DataFrame(d)
        return df

    @staticmethod
    def get_filepath(outdir: str, fname=OPTIMIZED_PARAMS_JSON) -> str:
        return os.path.join(outdir, fname)

    def _repr_html_(self):
        return self.to_dataframe()._repr_html_()
