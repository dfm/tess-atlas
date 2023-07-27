from __future__ import annotations

import logging
from typing import Dict, List, Union

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
from matplotlib.ticker import MaxNLocator, NullLocator

from tess_atlas.logger import LOGGER_NAME

from ..plotting_utils import (
    exception_catcher,
    format_hist_axes_label_string_with_offset,
    get_one_dimensional_median_and_error_bar,
)

logger = logging.getLogger(LOGGER_NAME)


def get_range(data: pd.DataFrame, params: List[str]) -> List[List[int]]:
    """Gets bounds of dataset"""
    if isinstance(data, pd.DataFrame):
        return [[data[p].min(), data[p].max()] for p in params]
    elif isinstance(data, az.InferenceData):
        data_array = __convert_to_numpy_list(data, params)
        return [[min(d), max(d)] for d in data_array]
    else:
        raise TypeError("Unexpected type provided to get_range")


def make_titles(df):
    titles = []
    for c in list(df.columns):
        titles.append(get_one_dimensional_median_and_error_bar(df[c].values))
    return titles


def reformat_trues(p: Dict, keys: List[str], val_id: int) -> np.array:
    return np.array([p[k][val_id] for k in keys])


@exception_catcher
def plot_corner(
    data: Union[pd.DataFrame, az.InferenceData], extras: Dict
) -> plt.Figure:
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
    __add_quantile_titles(fig, kwargs)
    __format_ticks(fig)
    __format_tick_offset(fig)
    if kwargs.get("debug", False):
        __add_ax_num_on_corner_for_debugging(fig)

    return fig


def __add_ax_num_on_corner_for_debugging(fig):
    for i, ax in enumerate(fig.get_axes()):
        ax.annotate(
            f"{i}", (0.5, 0.5), xycoords=("axes fraction"), fontsize=30
        )


def __format_tick_offset(fig):
    axes = fig.get_axes()
    num_col = int(np.sqrt(len(axes) + 1))
    x_axes_to_format = [i + (num_col - 1) * num_col for i in range(num_col)]
    y_axes_to_format = [i * num_col for i in range(1, num_col)]
    both_to_format = (num_col - 1) * num_col
    for i in x_axes_to_format:
        format_hist_axes_label_string_with_offset(axes[i], "x")
        axes[i].set_yticklabels([])
    for i in y_axes_to_format:
        format_hist_axes_label_string_with_offset(axes[i], "y")
        axes[i].set_xticklabels([])
    format_hist_axes_label_string_with_offset(axes[both_to_format], "both")


def __format_ticks(fig, max_n_ticks=2):
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


def __add_quantile_titles(fig, kwargs):
    titles = kwargs["titles"]
    if len(titles) == 0:
        return
    axes = fig.get_axes()
    for i, title in enumerate(titles):
        ax = axes[i + i * len(titles)]
        ax.set_title(title, **kwargs["title_kwargs"])


def __convert_to_numpy_list(
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
