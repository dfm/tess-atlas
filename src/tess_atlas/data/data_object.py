import os
from abc import ABC
import logging

from tess_atlas.utils import NOTEBOOK_LOGGER_NAME
from .data_utils import sizeof

logger = logging.getLogger(NOTEBOOK_LOGGER_NAME)


class DataObject(ABC):
    @classmethod
    def load(cls, **kwargs):
        outdir = kwargs["outdir"]
        if cls.cached_data_present(cls.get_filepath(outdir)):
            try:
                return cls.from_cache(**kwargs)
            except Exception as e:
                logger.info(f"Error loading fom cache: ''{e}''")
        return cls.from_database(**kwargs)

    @classmethod
    def from_database(cls, **kwargs):
        raise NotImplementedError()

    @classmethod
    def from_cache(cls, **kwargs):
        raise NotImplementedError()

    def save_data(self):
        raise NotImplementedError()

    @staticmethod
    def get_filepath(outdir: str, fname: str) -> str:
        return os.path.join(outdir, fname)

    @staticmethod
    def cached_data_present(fpath: str) -> bool:
        return os.path.isfile(fpath)

    @property
    def mem_size(self):
        return sizeof(self)
