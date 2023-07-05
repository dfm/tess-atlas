import os
import shutil
import unittest

import pandas as pd
import pytest

from tess_atlas.data import TICEntry

CLEAN_AFTER_TEST = False


@unittest.skip("This is a test for a bug still being investigated")
class TestSingleTransitSetter(unittest.TestCase):
    def setUp(self):
        norm_tois = [103, 978]
        single_trans_tois = [5153, 2168]
        self.all_tois = norm_tois + single_trans_tois
        self.norm_tois = [TICEntry.load(toi) for toi in norm_tois]
        self.single_trans_tois = [
            TICEntry.load(toi) for toi in single_trans_tois
        ]

    def tearDown(self):
        if CLEAN_AFTER_TEST:
            for toi in self.all_tois:
                shutil.rmtree(f"toi_{toi}_files")

    def test_norm_transits_single_transit_flag(self):
        for tic in self.norm_tois:
            for pi, p in enumerate(tic.candidates):
                id = f"TOI{tic.toi_number}.{pi+1}"
                self.assertFalse(
                    p.has_data_only_for_single_transit,
                    f"{id} incorrectly labeled as single-transit system",
                )

    def test_norm_transits_num_period(self):
        for tic in self.norm_tois:
            for pi, p in enumerate(tic.candidates):
                id = f"TOI{tic.toi_number}.{pi+1}"
                self.assertGreater(
                    p.num_periods, 0, f"{id} num periods {p.num_periods} < 0"
                )

    def test_norm_transits_tmin_tmax(self):
        for tic in self.norm_tois:
            for pi, p in enumerate(tic.candidates):
                id = f"TOI{tic.toi_number}.{pi+1}"
                self.assertNotEqual(
                    p.tmin, p.tmax, f"{id} tmin==tmax for non-single transit!"
                )

    def test_single_transits_single_transit_flag(self):
        for tic in self.single_trans_tois:
            for pi, p in enumerate(tic.candidates):
                id = f"TOI{tic.toi_number}.{pi+1}"
                self.assertTrue(
                    p.has_data_only_for_single_transit,
                    f"{id} incorrectly labeled as norm system",
                )

    def test_single_transits_num_period(self):
        for tic in self.single_trans_tois:
            for pi, p in enumerate(tic.candidates):
                id = f"TOI{tic.toi_number}.{pi+1}"
                self.assertEqual(
                    p.num_periods, 0, f"{id} num periods {p.num_periods} != 0"
                )

    def test_single_transits_tmin_tmax(self):
        for tic in self.single_trans_tois:
            for pi, p in enumerate(tic.candidates):
                id = f"TOI{tic.toi_number}.{pi+1}"
                self.assertEqual(
                    p.tmin, p.tmax, f"{id} tmin!=tmax for single transit!"
                )


if __name__ == "__main__":
    unittest.main()
