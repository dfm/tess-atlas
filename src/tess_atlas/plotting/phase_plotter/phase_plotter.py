from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Optional

import matplotlib.pyplot as plt
from arviz import InferenceData
from pymc3 import Model

from ...logger import LOGGER_NAME
from .core import _preprocess_phase_plot_data, add_phase_data_to_ax

if TYPE_CHECKING:
    from ...data import TICEntry

logger = logging.getLogger(LOGGER_NAME)


def plot_phase(
    tic_entry: "TICEntry",
    model: Model,
    inference_data: Optional[InferenceData] = None,
    initial_params: Optional[Dict] = None,
    **kwgs,
):
    """Adapted from exoplanet tutorials
    https://gallery.exoplanet.codes/tutorials/transit/#phase-plots

    In addition to {tic_entry, model} this function needs either
    - inference_data
    - initial_params
    """

    kwgs = _preprocess_phase_plot_data(
        model, inference_data, initial_params, kwgs
    )

    if kwgs.get("thumbnail", False):
        kwgs.update(
            data_bins=80,
            default_fs=0,
            period_fs=0,
            legend_fs=0,
            ms=2.5,
            savekwgs=dict(
                transparent=True, dpi=80, bbox_inches="tight", pad_inches=0
            ),
            annotate_with_period=False,
            figsize=(2.0, 1.5),
            legend=0,
        )
    figsize = kwgs.get("figsize", (7, 5))

    for i in range(tic_entry.planet_count):
        fig = plt.figure(figsize=figsize)
        add_phase_data_to_ax(fig.gca(), i, tic_entry, **kwgs)
