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
        data = tess_data.TICEntry.generate_tic_from_toi_number(toi=103)
        self.assertIsInstance(data.to_dataframe(), pd.DataFrame)

    def test_exofop_csv_getter(self):
        db = tess_data.tic_entry.get_tic_database()
        self.assertIsInstance(db, pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
