import os
import shutil
import unittest

import pandas as pd

import tess_atlas.data as tess_data
from tess_atlas.utils import NOTEBOOK_LOGGER_NAME, setup_logger

CLEAN_AFTER_TEST = True


class TestData(unittest.TestCase):
    def setUp(self):
        self.orig_dir = os.getcwd()
        self.outdir = "data_test_outdir"
        os.makedirs(self.outdir, exist_ok=True)
        os.chdir(self.outdir)
        self.logger = setup_logger(NOTEBOOK_LOGGER_NAME, outdir=self.outdir)

    def tearDown(self):
        os.chdir(self.orig_dir)
        if os.path.exists(self.outdir) and CLEAN_AFTER_TEST:
            shutil.rmtree(self.outdir)

    def test_data_download(self):
        self.logger.info("LOADING FROM INTERNET")
        data = tess_data.TICEntry.load(toi=103)
        self.assertIsInstance(data.to_dataframe(), pd.DataFrame)
        self.assertFalse(data.loaded_from_cache)
        self.logger.info("LOADING FROM CACHE")
        data = tess_data.TICEntry.load(toi=103)
        self.assertTrue(data.loaded_from_cache)


if __name__ == "__main__":
    unittest.main()
