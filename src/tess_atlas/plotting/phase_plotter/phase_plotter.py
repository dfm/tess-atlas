from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Optional

import matplotlib.pyplot as plt
from arviz import InferenceData
from pymc3 import Model

from tess_atlas.logger import LOGGER_NAME

from .core import _preprocess_phase_plot_data, add_phase_data_to_ax

if TYPE_CHECKING:
    from tess_atlas.data.tic_entry import TICEntry

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

    figsize = kwgs.get("figsize", (7, 5))
    for i in range(tic_entry.planet_count):
        fig = plt.figure(figsize=figsize)
        add_phase_data_to_ax(fig.gca(), i, tic_entry, **kwgs)
