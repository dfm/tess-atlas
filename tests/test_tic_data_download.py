import pandas as pd

import tess_atlas.data as tess_data
from tess_atlas.utils import NOTEBOOK_LOGGER_NAME, setup_logger


def test_data_download(tmp_path):
    """TODO: mock this test as downloading data can take a while"""
    logger = setup_logger(NOTEBOOK_LOGGER_NAME, outdir=tmp_path)
    logger.info("LOADING FROM INTERNET")
    data = tess_data.TICEntry.load(toi=103)
    assert isinstance(data.to_dataframe(), pd.DataFrame)
    assert data.loaded_from_cache == False
    logger.info("LOADING FROM CACHE")
    data = tess_data.TICEntry.load(toi=103)
    assert data.loaded_from_cache
