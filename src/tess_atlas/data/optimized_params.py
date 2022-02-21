import logging
import os
from typing import Dict, List

from tess_atlas.utils import NOTEBOOK_LOGGER_NAME

from .data_object import DataObject
from .data_utils import load_json, save_json

logger = logging.getLogger(NOTEBOOK_LOGGER_NAME)


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

    def to_dict(self):
        return self.params

    @staticmethod
    def get_filepath(outdir: str, fname="optimized_params.json") -> str:
        return os.path.join(outdir, fname)
