import logging
import os
from typing import Dict, List, Optional

import corner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator, NullLocator, ScalarFormatter

from tess_atlas.data import TICEntry
from tess_atlas.data.inference_data_tools import get_samples_dataframe
from tess_atlas.utils import NOTEBOOK_LOGGER_NAME

from .labels import ECCENTRICITY_PLOT, LATEX, POSTERIOR_PLOT, PRIOR_PLOT
from .plotting_utils import (
    exception_catcher,
    format_label_string_with_offset,
    format_prior_samples_and_initial_params,
    get_colors,
    get_one_dimensional_median_and_error_bar,
    get_range,
)

logger = logging.getLogger(NOTEBOOK_LOGGER_NAME)


@exception_catcher
def plot_corner(data, extras):
    kwargs = dict(
        smooth=0.9,
        label_kwargs=dict(fontsize=20),
        labelpad=-0.04,
        title_kwargs=dict(fontsize=20, pad=10),
        color="#0072C1",
        truth_color="tab:red",
        quantiles=[0.16, 0.84],
        levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9.0 / 2.0)),
        plot_density=False,
        plot_datapoints=False,
        fill_contours=True,
        max_n_ticks=2,
        verbose=False,
        use_math_text=True,
        titles=[],
    )
    kwargs.update(extras)

    fig = corner.corner(data, **kwargs)
    add_quantile_titles(fig, kwargs)
    format_ticks(fig)
    format_tick_offset(fig, kwargs)
    # _add_ax_num_on_corner_for_debugging(fig)

    return fig


def _add_ax_num_on_corner_for_debugging(fig):
    for i, ax in enumerate(fig.get_axes()):
        ax.annotate(
            f"{i}", (0.5, 0.5), xycoords=("axes fraction"), fontsize=30
        )


def format_tick_offset(fig, kwargs):
    axes = fig.get_axes()
    num_col = int(np.sqrt(len(axes) + 1))
    x_axes_to_format = [i + (num_col - 1) * num_col for i in range(num_col)]
    y_axes_to_format = [i * num_col for i in range(1, num_col)]
    both_to_format = (num_col - 1) * num_col
    for i in x_axes_to_format:
        format_label_string_with_offset(axes[i], "x")
        axes[i].set_yticklabels([])
    for i in y_axes_to_format:
        format_label_string_with_offset(axes[i], "y")
        axes[i].set_xticklabels([])
    format_label_string_with_offset(axes[both_to_format], "both")


def format_ticks(fig, max_n_ticks=2):
    axes = fig.get_axes()
    nax = len(axes)
    num_row = num_col = int(np.sqrt(nax))
    hist1ds = [i + num_col * i for i in range(num_col)]
    hist2ds = []
    for i in range(num_row):
        for j in range(i * num_col, hist1ds[i]):
            hist2ds.append(j)

    for hist1d_id in hist1ds:
        ax = axes[hist1d_id]
        ax.xaxis.set_major_locator(MaxNLocator(max_n_ticks, prune="both"))
        ax.yaxis.set_major_locator(NullLocator())

    for hist2d_id in hist2ds:
        ax = axes[hist2d_id]
        ax.xaxis.set_major_locator(MaxNLocator(max_n_ticks, prune="both"))
        ax.yaxis.set_major_locator(MaxNLocator(max_n_ticks, prune="both"))


def add_quantile_titles(fig, kwargs):
    titles = kwargs["titles"]
    if len(titles) == 0:
        return
    axes = fig.get_axes()
    for i, title in enumerate(titles):
        ax = axes[i + i * len(titles)]
        ax.set_title(title, **kwargs["title_kwargs"])


def make_titles(df):
    titles = []
    for c in list(df.columns):
        titles.append(get_one_dimensional_median_and_error_bar(df[c].values))
    return titles


def reformat_trues(p: Dict, keys: List[str], val_id: int) -> np.array:
    return np.array([p[k][val_id] for k in keys])


@exception_catcher
def plot_posteriors(
    tic_entry: TICEntry,
    inference_data,
    initial_params: Optional[Dict] = {},
    plot_params=[],
    title=True,
    fname="",
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

    if len(fname) == 0:
        fname = os.path.join(
            tic_entry.outdir, f"planet_{'{}'}_{POSTERIOR_PLOT}"
        )
    colors = get_colors(tic_entry.planet_count)
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
            plt.suptitle(f"TOI {tic_entry.toi_number}\nPlanet {n+1} Posterior")
        logger.debug(f"Saving {fname.format(n+1)}")
        fig.savefig(fname.format(n + 1), bbox_inches="tight")


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
    fig.savefig(fname, bbox_inches="tight")


@exception_catcher
def plot_eccentricity_posteriors(
    tic_entry: TICEntry, ecc_samples: pd.DataFrame, title=True, fname=""
) -> None:
    params = ["e", "omega"]
    colors = get_colors(tic_entry.planet_count)
    if len(fname) == 0:
        fname = os.path.join(
            tic_entry.outdir, f"planet_{'{}'}_{ECCENTRICITY_PLOT}"
        )
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
            plt.suptitle(f"Planet {n}", x=0.85, y=0.85, va="top", ha="right")
        logger.debug(f"Saving {fname.format(n+1)}")
        fig.savefig(fname.format(n + 1), bbox_inches="tight")
