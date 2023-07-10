import pandas as pd


class Catalog:
    """Class that reads in all the catalog PE data from the InferenceData objects"""

    def __init__(self, catalog_data: pd.DataFrame):
        self.__data = catalog_data

    @classmethod
    def from_dir(self, dir: str):
        pass

    @classmethod
    def from_cache(self, cache_dir: str):
        pass
