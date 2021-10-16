import os
import shutil
import unittest

import pandas as pd

import tess_atlas.data as tess_data

CLEAN_AFTER_TEST = True


class TestData(unittest.TestCase):
    def setUp(self):
        self.orig_dir = os.getcwd()
        self.outdir = "data_test_outdir"
        os.makedirs(self.outdir, exist_ok=True)
        os.chdir(self.outdir)

    def tearDown(self):
        os.chdir(self.orig_dir)
        if os.path.exists(self.outdir) and CLEAN_AFTER_TEST:
            shutil.rmtree(self.outdir)

    def test_data_download(self):
        data = tess_data.TICEntry.load_tic_data(toi=103)
        self.assertIsInstance(data.to_dataframe(), pd.DataFrame)
        self.assertFalse(data.loaded_from_cache)
        data = tess_data.TICEntry.load_tic_data(toi=103)
        self.assertTrue(data.loaded_from_cache)


if __name__ == "__main__":
    unittest.main()
