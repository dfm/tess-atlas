from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Dict, Optional

import matplotlib.pyplot as plt
from arviz import InferenceData
from pymc3 import Model

from tess_atlas.logger import LOGGER_NAME

from ..labels import THUMBNAIL_PLOT
from ..plotting_utils import get_colors
from .core import _preprocess_phase_plot_data, add_phase_data_to_ax

if TYPE_CHECKING:
    from tess_atlas.data.tic_entry import TICEntry

logger = logging.getLogger(LOGGER_NAME)


def plot_thumbnail(
    tic_entry: "TICEntry",
    model: Model,
    inference_data: Optional[InferenceData] = None,
    initial_params: Optional[Dict] = None,
    **kwgs,
):
    kwgs = _preprocess_phase_plot_data(
        model, inference_data, initial_params, kwgs
    )
    kwgs.update(
        data_bins=80,
        default_fs=0,
        period_fs=0,
        legend_fs=0,
        annotate_with_period=False,
        legend=0,
        lc_alpha=0.5,
        lc_fill_alpha=0.25,
    )
    fig = plt.figure(figsize=(2.0, 1.5))
    ax = fig.gca()
    colors = get_colors(tic_entry.planet_count)
    for i in range(tic_entry.planet_count):
        kwgs.update(
            binned_data_kwgs=dict(ms=2.5, color=colors[i], alpha=0.25),
        )
        add_phase_data_to_ax(ax, i, tic_entry, **kwgs)
    ax.set_title("")
    plt.axis("off")
    plt.margins(x=0, y=0.1, tight=True)
    plt.xlabel("")
    plt.ylabel("")
    plt.axis("tight")
    fname = os.path.join(tic_entry.outdir, THUMBNAIL_PLOT)
    fig.savefig(
        fname, transparent=True, dpi=80, bbox_inches="tight", pad_inches=0
    )
