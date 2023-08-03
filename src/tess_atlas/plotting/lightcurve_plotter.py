from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tess_atlas.data.tic_entry import TICEntry

from ..logger import LOGGER_NAME
from .labels import FLUX_LABEL, LIGHTCURVE_PLOT, TIME_LABEL
from .plotting_utils import get_colors, get_longest_unbroken_section_of_data

logger = logging.getLogger(LOGGER_NAME)


def plot_lightcurve(
    tic_entry: TICEntry,
    model_lightcurves: Optional[np.ndarray] = None,
    save: Optional[bool] = True,
    zoom_in: Optional[bool] = False,
    observation_periods: Optional[np.ndarray] = None,
) -> Tuple[plt.Figure, None]:
    """Plot lightcurve data"""
    # TODO: truncate region of missing data on axes

    if model_lightcurves is None:
        model_lightcurves = []
    else:
        model_lightcurves = model_lightcurves.T

    if observation_periods is None:
        observation_periods = []

    colors = get_colors(tic_entry.planet_count)
    fig, ax = plt.subplots(1, figsize=(9, 5))

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
    plt.tight_layout()
    if save:
        logger.info(f"Saving {fname}")
        fig.savefig(fname)
        plt.close(fig)
    else:
        return fig


def plot_interactive_lightcurve(
    tic_entry: "TICEntry", model_lightcurves: Optional[List[float]] = None
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
