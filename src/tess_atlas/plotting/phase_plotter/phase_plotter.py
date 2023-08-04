from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
from arviz import InferenceData
from pymc3 import Model

from tess_atlas.logger import LOGGER_NAME

from ..image_utils import vertical_image_concat
from ..labels import PHASE_PLOT
from .core import _preprocess_phase_plot_data, add_phase_data_to_ax

if TYPE_CHECKING:
    from tess_atlas.data.tic_entry import TICEntry

logger = logging.getLogger(LOGGER_NAME)


def plot_phase(
    tic_entry: "TICEntry",
    model: Model,
    inference_data: Optional[InferenceData] = None,
    initial_params: Optional[Dict] = None,
    save=Union[bool, str],
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
    toi = f"TOI{tic_entry.toi_number}"
    fnames = []
    for i in range(tic_entry.planet_count):
        fig = plt.figure(figsize=(7, 5))
        add_phase_data_to_ax(fig.gca(), i, tic_entry, **kwgs)
        fname = PHASE_PLOT.replace(".", f"_{toi}_{i + 1}.")
        fname = os.path.join(tic_entry.outdir, fname)
        plt.savefig(fname, transparent=False, dpi=150)
        plt.close(fig)
        fnames.append(fname)
        logger.info(f"Saved {fname}")
    fname = save if isinstance(save, str) else PHASE_PLOT
    fpath = os.path.join(tic_entry.outdir, fname)
    vertical_image_concat(fnames, fpath)
    logger.info(f"Saved {fpath}")
