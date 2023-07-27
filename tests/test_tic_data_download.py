import pandas as pd

import tess_atlas.data.tic_entry as tess_data
from tess_atlas.logger import LOGGER_NAME, setup_logger


def test_data_download(tmp_working_dir):
    logger = setup_logger(LOGGER_NAME, outdir=tmp_working_dir)
    logger.info("LOADING FROM INTERNET")
    data = tess_data.TICEntry.load(toi=103, clean=True)
    assert isinstance(data.to_dataframe(), pd.DataFrame)
    assert data.loaded_from_cache == False
    logger.info("LOADING FROM CACHE")
    data = tess_data.TICEntry.load(toi=103)
    assert data.loaded_from_cache
