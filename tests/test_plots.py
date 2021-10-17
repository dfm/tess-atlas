import os
import shutil
import unittest

from tess_atlas.data.tic_entry import TICEntry

DATA_DIR = "test_data/toi_103_files"

CLEAR_AFTER_TEST = True


class MatplotlibPlotTest(unittest.TestCase):
    def setUp(self) -> None:
        self.out = "test_plotout"
        self.orig_dir = os.getcwd()
        os.makedirs(self.out, exist_ok=True)
        shutil.copytree(
            src=DATA_DIR,
            dst=os.path.join(self.out, os.path.basename(DATA_DIR)),
            dirs_exist_ok=True,
        )
        os.chdir(self.out)
        self.tic_entry = TICEntry.load(toi=103)

    def tearDown(self) -> None:
        if CLEAR_AFTER_TEST:
            shutil.rmtree(self.out)
        os.chdir(self.orig_dir)

    def test_static_phase_plot(self):
        os.environ["INTERACTIVE_PLOTS"] = "FALSE"
        from tess_atlas.plotting import plot_phase

        plot_phase(tic_entry=self.tic_entry)


if __name__ == "__main__":
    unittest.main()
