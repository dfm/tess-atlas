import logging
import os

import corner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm

from tess_atlas.data import TICEntry
from .labels import POSTERIOR_PLOT, ECCENTRICITY_PLOT

CORNER_KWARGS = dict(
    smooth=0.9,
    label_kwargs=dict(fontsize=30),
    title_kwargs=dict(fontsize=16),
    color="#0072C1",
    truth_color="tab:orange",
    quantiles=[0.16, 0.84],
    levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9.0 / 2.0)),
    plot_density=False,
    plot_datapoints=False,
    fill_contours=True,
    max_n_ticks=3,
    verbose=False,
    use_math_text=True,
)


def plot_posteriors(
    tic_entry: TICEntry, trace: pm.sampling.MultiTrace
) -> None:
    samples = pm.trace_to_dataframe(trace, varnames=["p", "r", "b"])
    fig = corner.corner(samples, **CORNER_KWARGS, range=get_range(samples))
    fname = os.path.join(tic_entry.outdir, POSTERIOR_PLOT)
    logging.debug(f"Saving {fname}")
    fig.savefig(fname)


def plot_eccentricity_posteriors(
    tic_entry: TICEntry, ecc_samples: pd.DataFrame
) -> None:
    for n in range(tic_entry.planet_count):
        planet_n_samples = ecc_samples[[f"e[{n}]", f"omega[{n}]"]]
        fig = corner.corner(
            planet_n_samples,
            weights=ecc_samples[f"weights[{n}]"],
            labels=["eccentricity", "omega"],
            **CORNER_KWARGS,
            range=get_range(planet_n_samples),
        )
        plt.suptitle(f"Planet {n} Eccentricity")
        fname = os.path.join(
            tic_entry.outdir, f"planet_{n}_{ECCENTRICITY_PLOT}"
        )
        logging.debug(f"Saving {fname}")
        fig.savefig(fname)


def get_range(samples):
    return [[samples[l].min(), samples[l].max()] for l in samples]
