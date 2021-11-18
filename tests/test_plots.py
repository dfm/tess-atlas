import os
import shutil
import unittest

import numpy as np
import pandas as pd

from tess_atlas.data.tic_entry import TICEntry

DATA_DIR = "test_data/toi_103_files"

CLEAR_AFTER_TEST = False


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

    def test_histogram_plot(self):
        from tess_atlas.plotting.histogram_plotter import plot_histograms

        trues = dict(a=0, b=5, c=2, d=-1)
        n = 5000
        samples = dict(
            a=np.random.uniform(
                low=trues["a"] - 1, high=trues["a"] + 1, size=n
            ),
            b=np.random.normal(loc=trues["b"], scale=10, size=n),
            c=np.random.pareto(a=trues["c"], size=n),
            d=np.random.standard_cauchy(size=n),
        )
        samples_table = {
            "Type 1": dict(a=samples["a"]),
            "Type 2": dict(b=samples["b"], c=samples["c"]),
            "Type 3": dict(d=samples["d"]),
        }
        plot_histograms(
            samples_table=samples_table, trues=trues, fname=f"./hist.png"
        )

        latex_label = dict(a=r"$\alpha$", b=r"$\beta$", c=r"$c$", d="d")
        plot_histograms(
            samples_table=samples_table,
            fname=f"./hist_latex.png",
            latex_label=latex_label,
        )


if __name__ == "__main__":
    unittest.main()
