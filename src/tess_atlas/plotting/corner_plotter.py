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

from .labels import ECCENTRICITY_PLOT, LATEX, POSTERIOR_PLOT, PRIOR_PLOT
from .plotting_utils import format_prior_samples_and_initial_params

logger = logging.getLogger(NOTEBOOK_LOGGER_NAME)

from .labels import ECCENTRICITY_PLOT, POSTERIOR_PLOT

CORNER_KWARGS = dict(
    smooth=0.9,
    label_kwargs=dict(fontsize=20),
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


def plot_posteriors(tic_entry: TICEntry, inference_data) -> None:
    params = ["r", "b", "t0", "tmax", "p", "dur"]
    fig = corner.corner(
        inference_data,
        var_names=params,
        **CORNER_KWARGS,
        range=get_range(inference_data, params),
        labels=[LATEX[p] for p in params],
    )
    fname = os.path.join(tic_entry.outdir, POSTERIOR_PLOT)
    logger.debug(f"Saving {fname}")
    fig.savefig(fname)


def plot_priors(
    tic_entry: TICEntry, prior_samples: Dict, init_params: Dict
) -> None:
    prior_samples, init_params = format_prior_samples_and_initial_params(
        prior_samples, init_params
    )
    fig = corner.corner(prior_samples, **CORNER_KWARGS, truths=init_params)
    fname = os.path.join(tic_entry.outdir, f"{PRIOR_PLOT}")
    logger.debug(f"Saving {fname}")
    fig.savefig(fname)


def plot_eccentricity_posteriors(
    tic_entry: TICEntry, ecc_samples: pd.DataFrame
) -> None:
    for n in range(tic_entry.planet_count):
        params = [f"e[{n}]", f"omega[{n}]"]
        planet_n_samples = ecc_samples[params]
        fig = corner.corner(
            planet_n_samples,
            weights=ecc_samples[f"weights[{n}]"],
            labels=["eccentricity", "omega"],
            **CORNER_KWARGS,
            range=get_range(planet_n_samples, params),
        )
        plt.suptitle(f"Planet {n} Eccentricity")
        fname = os.path.join(
            tic_entry.outdir, f"planet_{n}_{ECCENTRICITY_PLOT}"
        )
        logger.debug(f"Saving {fname}")
        fig.savefig(fname)


def get_range(data, params: List[str]) -> List[List[int]]:
    if isinstance(data, pd.DataFrame):
        return [[data[p].min(), data[p].max()] for p in params]
    elif isinstance(data, az.InferenceData):
        data_array = convert_to_numpy_list(data, params)
        return [[min(d), max(d)] for d in data_array]
    else:
        raise TypeError("Unexpected type provided to get_range")


def convert_to_numpy_list(
    inference_data: az.InferenceData, params: List[str]
) -> np.ndarray:
    """Converts from az.InferenceData --> 2D np.ndarray

    List[posterior_param_list_1, posterior_param_list_2...]
    each item in list is a list for the specific param
    """
    dataset = convert_to_dataset(inference_data, group="posterior")
    var_names = _var_names(params, dataset)
    plotters = list(
        xarray_var_iter(
            get_coords(dataset, {}), var_names=var_names, combined=True
        )
    )
    return np.stack([x[-1].flatten() for x in plotters], axis=0)
