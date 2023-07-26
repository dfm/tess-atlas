import os
import shutil
import unittest

import numpy as np
import pytest

from tess_atlas.data.tic_entry import TICEntry

DATA_DIR = "test_data/toi_103_files"

CLEAR_AFTER_TEST = False


@pytest.fixture
def tic_entry():
    return TICEntry.load(toi=103)


@pytest.mark.skip("Dont have a fast way to plug a model in here.")
def test_static_phase_plot(tic_entry):
    from tess_atlas.plotting import plot_phase

    plot_phase(tic_entry=tic_entry)


@pytest.mark.skip("No test data (data deleted).")
def test_histogram_plot():
    from tess_atlas.plotting.histogram_plotter import plot_histograms

    trues = dict(a=0, b=5, c=2, d=-1)
    n = 5000
    samples = dict(
        a=np.random.uniform(low=trues["a"] - 1, high=trues["a"] + 1, size=n),
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
