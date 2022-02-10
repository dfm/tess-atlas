import logging
import os
from typing import Dict, List

import arviz as az
import corner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from corner.arviz_corner import (
    _var_names,
    convert_to_dataset,
    get_coords,
    xarray_var_iter,
)
from pymc3.model import Model

from tess_atlas.analysis import get_untransformed_varnames, sample_prior
from tess_atlas.data import TICEntry
from tess_atlas.utils import NOTEBOOK_LOGGER_NAME
from tess_atlas.data.inference_data_tools import get_samples_dataframe

from .labels import ECCENTRICITY_PLOT, LATEX, POSTERIOR_PLOT, PRIOR_PLOT
from .plotting_utils import (
    format_prior_samples_and_initial_params,
    get_range,
    get_one_dimensional_median_and_error_bar,
    format_label_string_with_offset,
    get_colors,
)

logger = logging.getLogger(NOTEBOOK_LOGGER_NAME)

from .labels import ECCENTRICITY_PLOT, POSTERIOR_PLOT


def plot_corner(data, extras):
    kwargs = dict(
        smooth=0.9,
        label_kwargs=dict(fontsize=20),
        labelpad=0.1,
        title_kwargs=dict(fontsize=20),
        color="#0072C1",
        truth_color="tab:orange",
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
    format_tick_offset(fig, kwargs)

    return fig


def format_tick_offset(fig, kwargs):
    axes = fig.get_axes()
    num_col = int(np.sqrt(len(axes) + 1))
    x_axes_to_format = [
        i + (num_col - 1) * num_col for i in range(num_col - 1)
    ]
    y_axes_to_format = [i * num_col for i in range(1, num_col)]
    both_to_format = (num_col - 1) * num_col
    for i in x_axes_to_format:
        format_label_string_with_offset(axes[i], "x")
        axes[i].set_yticklabels([])
    for i in y_axes_to_format:
        format_label_string_with_offset(axes[i], "y")
        axes[i].set_xticklabels([])
    format_label_string_with_offset(axes[both_to_format], "both")


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


def plot_posteriors(tic_entry: TICEntry, inference_data) -> None:
    """Plots 1 posterior corner plot for each planet"""
    params = ["r", "b", "t0", "tmax", "p", "dur"]
    posterior_samples = get_samples_dataframe(inference_data)
    colors = get_colors(tic_entry.planet_count)
    for n in range(tic_entry.planet_count):
        planet_params = [f"{p}[{n}]" for p in params]
        planet_n_samples = posterior_samples[planet_params]
        fig = plot_corner(
            planet_n_samples,
            extras=dict(
                range=get_range(planet_n_samples, planet_params),
                labels=[f"{LATEX[p]}\n" for p in params],
                titles=make_titles(planet_n_samples),
                color=colors[n],
            ),
        )
        plt.suptitle(f"Planet {n} Posterior")
        fname = os.path.join(tic_entry.outdir, f"planet_{n}_{POSTERIOR_PLOT}")
        logger.debug(f"Saving {fname}")
        fig.savefig(fname)


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
    fig.savefig(fname)


def plot_priors(
    tic_entry: TICEntry, prior_samples: Dict, init_params: Dict
) -> None:
    prior_samples, init_params = format_prior_samples_and_initial_params(
        prior_samples, init_params
    )
    fig = plot_corner(prior_samples, extras=dict(truths=init_params))
    fname = os.path.join(tic_entry.outdir, f"{PRIOR_PLOT}")
    logger.debug(f"Saving {fname}")
    fig.savefig(fname)


def plot_eccentricity_posteriors(
    tic_entry: TICEntry, ecc_samples: pd.DataFrame
) -> None:
    params = ["e", "omegea"]
    colors = get_colors(tic_entry.planet_count)
    for n in range(tic_entry.planet_count):
        planet_params = [f"{p}[{n}]" for p in params]
        planet_n_samples = ecc_samples[planet_params]
        fig = plot_corner(
            planet_n_samples,
            extras=dict(
                weights=ecc_samples[f"weights[{n}]"],
                labels=[LATEX[p] for p in params],
                range=get_range(planet_n_samples, planet_params),
                titles=make_titles(planet_n_samples),
                color=colors[n],
            ),
        )
        plt.suptitle(f"Planet {n} Eccentricity")
        fname = os.path.join(
            tic_entry.outdir, f"planet_{n}_{ECCENTRICITY_PLOT}"
        )
        logger.debug(f"Saving {fname}")
        fig.savefig(fname)
