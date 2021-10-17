import unittest
import os
import shutil
from tess_atlas.data.tic_entry import (
    TICEntry,
    LightCurveData,
    StellarData,
    InferenceData,
)


CLEAR_AFTER_TEST = False


class MatplotlibPlotTest(unittest.TestCase):
    def setUp(self) -> None:
        self.out = "test_plotout"
        os.makedirs(self.out, exist_ok=True)
        self.tic_entry = TICEntry.load(toi=103)

    def tearDown(self) -> None:
        if CLEAR_AFTER_TEST:
            shutil.rmtree(self.out)

    def test_static_phase_plot(self):
        os.environ["INTERACTIVE_PLOTS"] = "FALSE"
        from tess_atlas.plotting import plot_phase

        plot_phase(tic_entry=self.tic_entry)


if __name__ == "__main__":
    unittest.main()
