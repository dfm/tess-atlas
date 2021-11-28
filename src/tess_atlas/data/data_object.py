import os
from abc import ABC


class DataObject(ABC):
    @classmethod
    def load(cls, **kwargs):
        outdir = kwargs["outdir"]
        if cls.cached_data_present(cls.get_filepath(outdir)):
            return cls.from_cache(**kwargs)
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
