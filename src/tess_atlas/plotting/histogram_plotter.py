import logging
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker
from matplotlib.ticker import MaxNLocator, ScalarFormatter

from tess_atlas.data import TICEntry
from tess_atlas.utils import NOTEBOOK_LOGGER_NAME

from .labels import (
    ECCENTRICITY_PLOT,
    LATEX,
    PARAMS_CATEGORIES,
    POSTERIOR_PLOT,
    PRIOR_PLOT,
)
from .plotting_utils import format_prior_samples_and_initial_params

logger = logging.getLogger(NOTEBOOK_LOGGER_NAME)


def plot_priors(
    tic_entry: TICEntry, prior_samples: Dict, init_params: Dict
) -> None:
    prior_samples, init_params = format_prior_samples_and_initial_params(
        prior_samples, init_params
    )

    samples_table = {}
    samples_table["Noise Params"] = get_samples_from_param_regexs(
        prior_samples, PARAMS_CATEGORIES["NOISE PARAMS"]
    )
    samples_table["Stellar Params"] = get_samples_from_param_regexs(
        prior_samples, PARAMS_CATEGORIES["STELLAR PARAMS"]
    )
    samples_table[f"Period Params"] = get_samples_from_param_regexs(
        prior_samples, PARAMS_CATEGORIES["PERIOD PARAMS"]
    )
    samples_table[f"Planet Params"] = get_samples_from_param_regexs(
        prior_samples, PARAMS_CATEGORIES["PLANET PARAMS"]
    )

    try:
        fig = plot_histograms(
            samples_table, trues=init_params, latex_label=LATEX
        )

        fname = os.path.join(tic_entry.outdir, f"{PRIOR_PLOT}")
        logger.debug(f"Saving {fname}")
        fig.savefig(fname)
    except Exception as e:
        logger.error(f"Cant plot priors: {e}")


def get_samples_from_param_regexs(samples, param_regex):
    samples_keys = samples.columns.values
    data = {}
    for s in samples_keys:
        for regex in param_regex:
            if regex == s or f"{regex}_" in s:
                data[s] = samples[s]
    return data


def plot_histograms(
    samples_table: Dict[str, Dict[str, np.array]],
    fname: Optional[str] = "",
    trues: Optional[Dict] = {},
    latex_label: Optional[Dict] = {},
) -> None:
    nrows = len(samples_table.keys())
    ncols = __get_longest_row_length(samples_table)
    fig, axes = __create_fig(nrows, ncols)

    for row_i, (set_label, sample_set) in enumerate(samples_table.items()):
        axes[row_i, 0].set_title(set_label + ":", loc="left")
        for col_i, (sample_name, samples) in enumerate(sample_set.items()):
            __plot_hist1d(axes[row_i, col_i], samples)

            if trues:
                axes[row_i, col_i].axvline(trues[sample_name], color="C1")
            __add_ax(axes[row_i, col_i])
            axes[row_i, col_i].set_xlabel(
                latex_label.get(sample_name, sample_name)
            )
            format_label_string_with_offset(axes[row_i, col_i], "x")
    plt.tight_layout()
    if fname:
        fig.savefig(fname)
    else:
        return fig


def __plot_hist1d(ax, x):
    range = np.quantile(x, [0.01, 0.99])
    ax.hist(x, density=True, bins=20, range=range, histtype="step", color="C0")
    ax.set_xlim(min(range), max(range))


def __create_fig(nrows, ncols):
    xdim, ydim = ncols * 2.5, nrows * 2.5
    fig, axes = plt.subplots(nrows, ncols, figsize=(xdim, ydim))
    for i in range(nrows):
        for j in range(ncols):
            __remove_ax(axes[i, j])
    return fig, axes


def __remove_ax(ax):
    ax.set_yticks([])
    ax.set_xticks([])
    ax.tick_params(direction="in")
    ax.set_frame_on(False)


def __add_ax(ax):
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.tick_params(direction="in")
    ax.set_frame_on(True)


def update_label(old_label, offset_text):
    if offset_text == "":
        return old_label

    try:
        units = old_label[old_label.index("[") + 1 : old_label.rindex("]")]
    except ValueError:
        units = ""
    label = old_label.replace("[{}]".format(units), "")

    if "+" in offset_text:
        offset_text = "+" + str(int(float(offset_text.replace("+", ""))))

    return "{} [{} {}]".format(label, offset_text, units)


def format_label_string_with_offset(ax, axis="both"):
    """Format the label string with the exponent from the ScalarFormatter"""
    ax.ticklabel_format(axis=axis, style="sci", scilimits=(-1e4, 1e4))

    axes_instances = []
    if axis in ["x", "both"]:
        axes_instances.append(ax.xaxis)
    if axis in ["y", "both"]:
        axes_instances.append(ax.yaxis)

    for ax in axes_instances:
        ax.major.formatter._useMathText = False
        ax.major.formatter._useOffset = True

        plt.draw()  # Update the text
        offset_text = ax.get_offset_text().get_text()

        label = ax.get_label().get_text()
        ax.offsetText.set_visible(False)
        ax.set_label_text(update_label(label, offset_text))


def __get_longest_row_length(
    samples_table: Dict[str, Dict[str, np.array]]
) -> int:
    return max(
        [
            len(samples_dicts.keys())
            for label, samples_dicts in samples_table.items()
        ]
    )
