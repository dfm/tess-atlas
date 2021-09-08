import os
import shutil
import unittest
import tess_atlas.data as tess_data
import pandas as pd

CLEAN_AFTER_TEST = False

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


if __name__ == '__main__':
    unittest.main()
