from __future__ import annotations

import logging
import os
from typing import Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tess_atlas.data.inference_data_tools import get_samples_dataframe
from tess_atlas.data.tic_entry import TICEntry
from tess_atlas.logger import LOGGER_NAME

from ..image_utils import vertical_image_concat
from ..labels import ECCENTRICITY_PLOT, LATEX, POSTERIOR_PLOT, PRIOR_PLOT
from ..plotting_utils import (
    exception_catcher,
    format_prior_samples_and_initial_params,
    get_colors,
)
from .core import get_range, make_titles, plot_corner, reformat_trues

logger = logging.getLogger(LOGGER_NAME)


@exception_catcher
def plot_posteriors_all(tic_entry: TICEntry, inference_data) -> None:
    """Plots posteriors for all planets in one corner"""
    params = ["r", "b", "t0", "tmax", "p", "dur"]
    fig = plot_corner(
        inference_data,
        extras=dict(var_names=params, range=get_range(inference_data, params)),
    )
    fname = os.path.join(
        tic_entry.outdir, POSTERIOR_PLOT.replace(".png", "_all.png")
    )
    logger.debug(f"Saving {fname}")
    fig.savefig(fname, bbox_inches="tight")


def plot_priors(
    tic_entry: TICEntry, prior_samples: Dict, init_params: Dict
) -> None:
    prior_samples, init_params = format_prior_samples_and_initial_params(
        prior_samples, init_params
    )
    fig = plot_corner(prior_samples.values, extras=dict(truths=init_params))
    fname = os.path.join(tic_entry.outdir, f"{PRIOR_PLOT}")
    logger.debug(f"Saving {fname}")
    if fig is not None:
        fig.savefig(fname, bbox_inches="tight")


@exception_catcher
def plot_eccentricity_posteriors(
    tic_entry: TICEntry, ecc_samples: pd.DataFrame, title=True, save=""
) -> None:
    params = ["e", "omega"]
    colors = get_colors(tic_entry.planet_count)
    flabel = os.path.join(
        tic_entry.outdir, f"planet_{'{}'}_{ECCENTRICITY_PLOT}"
    )
    fnames = []
    for n in range(tic_entry.planet_count):
        planet_params = [f"{p}[{n}]" for p in params]
        planet_n_samples = ecc_samples[planet_params]
        fig = plot_corner(
            planet_n_samples.values,
            extras=dict(
                weights=ecc_samples[f"weights[{n}]"],
                labels=[LATEX[p] for p in params],
                range=get_range(planet_n_samples, planet_params),
                titles=make_titles(planet_n_samples),
                color=colors[n],
            ),
        )
        if title:
            plt.suptitle(f"Planet {n+1}", x=0.85, y=0.85, va="top", ha="right")

        if fig is not None:
            fname = flabel.format(n + 1)
            fig.savefig(flabel.format(n + 1), bbox_inches="tight")
            logger.info(f"Saved {fname}")
            fnames.append(fname)
        plt.close(fig)

    if save:
        savefname = save if isinstance(save, str) else ECCENTRICITY_PLOT
        savefname = os.path.join(tic_entry.outdir, savefname)
        vertical_image_concat(fnames, savefname)
        logger.info(f"Saved {savefname}")


@exception_catcher
def plot_posteriors(
    tic_entry: TICEntry,
    inference_data,
    initial_params: Optional[Dict] = {},
    plot_params=[],
    title=True,
    save=Union[bool, str],
) -> None:
    """Plots 1 posterior corner plot for each planet"""
    valid_params = [
        "p",
        "b",
        "r",
        "dur",
        "tmin",
        "tmax",
    ]

    if len(plot_params) == 0:
        plot_params = valid_params
    else:
        plot_params = [p for p in plot_params if p in valid_params]

    single_transit_params = ["log_p", "b", "log_r", "dur", "tmin"]

    if initial_params:
        initial_params["log_r"] = np.log(initial_params["r"])
        initial_params["log_p"] = np.log(initial_params["p"])

    posterior_samples = get_samples_dataframe(inference_data)

    flabel = os.path.join(tic_entry.outdir, f"planet_{'{}'}_{POSTERIOR_PLOT}")
    colors = get_colors(tic_entry.planet_count)
    fnames = []
    for n in range(tic_entry.planet_count):
        params = plot_params.copy()
        if tic_entry.candidates[n].has_data_only_for_single_transit:
            params = single_transit_params.copy()
        planet_params = [f"{p}[{n}]" for p in params]
        posterior_samples[f"log_r[{n}]"] = np.log(posterior_samples[f"r[{n}]"])
        posterior_samples[f"log_p[{n}]"] = np.log(posterior_samples[f"p[{n}]"])
        planet_n_samples = posterior_samples[planet_params]
        extras = dict(
            range=get_range(planet_n_samples, planet_params),
            labels=[f"{LATEX[p]}\n" for p in params],
            titles=make_titles(planet_n_samples),
            color=colors[n],
        )
        if initial_params:
            truths = reformat_trues(initial_params, params, n)
            if not np.isnan(truths).any():
                extras["truths"] = truths
        fig = plot_corner(planet_n_samples.values, extras=extras)
        if title:
            plt.suptitle(
                f"TOI {tic_entry.toi_number}\nPlanet {n + 1} Posterior"
            )
        if fig is not None:
            fname = flabel.format(n + 1)
            fig.savefig(fname, bbox_inches="tight")
            logger.info(f"Saved {fname}")
            fnames.append(fname)
            plt.close(fig)

    if len(fnames) != 0:
        fpath = POSTERIOR_PLOT if isinstance(save, bool) else save
        fpath = os.path.join(tic_entry.outdir, fpath)
        vertical_image_concat(fnames, fpath)
        logger.info(f"Saved combined {fpath}")
