import logging
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from arviz import InferenceData
from plotly.subplots import make_subplots

from tess_atlas.data import TICEntry
from tess_atlas.data.inference_data_tools import get_samples_dataframe
from tess_atlas.utils import NOTEBOOK_LOGGER_NAME

from .extra_plotting.ci import plot_ci, plot_xy_binned
from .labels import (
    FLUX_LABEL,
    LIGHTCURVE_PLOT,
    PHASE_PLOT,
    TIME_LABEL,
    TIME_SINCE_TRANSIT_LABEL,
)
from .plotting_utils import (
    fold_data,
    fold_lightcurve_models,
    generate_model_lightcurve,
    get_colors,
    get_lc_and_gp_from_inference_object,
    get_longest_unbroken_section_of_data,
)

logger = logging.getLogger(NOTEBOOK_LOGGER_NAME)


def plot_lightcurve(
    tic_entry: TICEntry,
    model_lightcurves: Optional[np.ndarray] = None,
    save: Optional[bool] = True,
    zoom_in: Optional[bool] = False,
    observation_periods: Optional[np.ndarray] = None,
) -> plt.Figure:
    """Plot lightcurve data + transit fits (if provided) in one plot

    model_lightcurves: (lightcurve_num, lightcurve_y_vals, planet_id)

    """
    # todo truncate region of missing data on axes

    if model_lightcurves is None:
        model_lightcurves = []
    else:
        model_lightcurves = model_lightcurves.T

    if observation_periods is None:
        observation_periods = []

    colors = get_colors(tic_entry.planet_count)
    fig, ax = plt.subplots(1, figsize=(7, 5))

    lc = tic_entry.lightcurve

    if zoom_in:
        idx, _ = get_longest_unbroken_section_of_data(lc.time)
        perc_data = int(100 * (len(idx) / len(lc.time)))
        logger.debug(f"{perc_data}% Data Displayed")
    else:
        idx = [i for i in range(len(lc.time))]

    ax.scatter(
        lc.time[idx],
        lc.flux[idx],
        color="k",
        label="Data",
        s=0.75,
        alpha=0.5,
    )
    ax.set_ylabel(FLUX_LABEL)
    ax.set_xlabel(TIME_LABEL)

    for i, model_lightcurve in enumerate(model_lightcurves):
        ax.plot(
            lc.time[idx],
            model_lightcurve[idx],
            label=f"Planet {i} fit",
            c=colors[i],
            alpha=0.75,
        )

    ax.legend(
        markerscale=5,
        frameon=False,
        bbox_to_anchor=(1.05, 1.0),
        loc="upper left",
    )

    for i, period in enumerate(observation_periods):
        c = "gray"
        if i % 2 == 0:
            c = "lightgray"
        for p in period:
            ax.axvline(
                p,
                color=c,
                ls="--",
                alpha=0.35,
                zorder=-100,
            )
        ax.axvspan(period[0], period[1], facecolor=c, alpha=0.1)

    fname = os.path.join(tic_entry.outdir, LIGHTCURVE_PLOT)
    if save:
        logger.debug(f"Saving {fname}")
        fig.savefig(fname)
    else:
        return fig


def plot_interactive_lightcurve(
    tic_entry: TICEntry, model_lightcurves: Optional[List[float]] = None
) -> go.Figure:
    """Plot lightcurve data + transit fits (if provided) in one plot"""
    if model_lightcurves is None:
        model_lightcurves = []
    lc = tic_entry.lightcurve
    fig = make_subplots(
        shared_xaxes=True,
        vertical_spacing=0.02,
        x_title=TIME_LABEL,
        y_title=FLUX_LABEL,
    )
    fig.add_trace(
        go.Scattergl(
            x=lc.time,
            y=lc.flux,
            mode="lines+markers",
            marker=dict(size=2, color="black"),
            line=dict(width=0.1),
            hoverinfo="skip",
            name="Data",
        )
    )
    for i, model_lightcurve in enumerate(model_lightcurves):
        fig.add_trace(
            go.Scattergl(
                x=lc.time,
                y=model_lightcurve,
                mode="lines",
                name=f"Planet {i}",
            )
        )
    fname = os.path.join(tic_entry.outdir, LIGHTCURVE_PLOT)
    logger.debug(f"Saving {fname}")
    fig.write_image(fname)
    return fig
