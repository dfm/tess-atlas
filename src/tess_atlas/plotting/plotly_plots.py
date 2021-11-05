import logging
import os
from typing import List, Optional

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tess_atlas.data import TICEntry
from tess_atlas.utils import NOTEBOOK_LOGGER_NAME

from .labels import (
    FLUX_LABEL,
    FOLDED_LIGHTCURVE_PLOT,
    LIGHTCURVE_PLOT,
    TIME_LABEL,
)
from .plotter_backend import PlotterBackend

logger = logging.getLogger(NOTEBOOK_LOGGER_NAME)


class PlotlyPlotter(PlotterBackend):
    @staticmethod
    def plot_lightcurve(
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

    @staticmethod
    def plot_folded_lightcurve(
        tic_entry: TICEntry, model_lightcurves: Optional[List[float]] = None
    ) -> go.Figure:
        """Subplots of folded lightcurves + transit fits (if provided) for each transit"""
        if model_lightcurves is None:
            model_lightcurves = []
        subplot_titles = [
            f"Planet {i + 1}: TOI-{c.toi_id}"
            for i, c in enumerate(tic_entry.candidates)
        ]
        fig = make_subplots(
            rows=tic_entry.planet_count,
            subplot_titles=subplot_titles,
            vertical_spacing=0.1,
        )
        for i in range(tic_entry.planet_count):
            lc = tic_entry.lightcurve
            planet = tic_entry.candidates[i]
            fig.add_trace(
                go.Scattergl(
                    x=planet.get_timefold(lc.time),
                    y=lc.flux,
                    mode="markers",
                    marker=dict(
                        size=3,
                        color=lc.time,
                        showscale=False,
                        colorbar=dict(title="Days"),
                    ),
                    name=f"Candidate {i + 1} Data",
                ),
                row=i + 1,
                col=1,
            )
            fig.update_xaxes(title_text=TIME_LABEL, row=i + 1, col=1)
            fig.update_yaxes(title_text=FLUX_LABEL, row=i + 1, col=1)
        for i, model_lightcurve in enumerate(model_lightcurves):
            lc = tic_entry.lightcurve
            planet = tic_entry.candidates[i]
            fig.add_trace(
                go.Scattergl(
                    x=planet.get_timefold(lc.time),
                    y=model_lightcurve,
                    mode="markers",
                    name=f"Planet {i + 1}",
                ),
                row=i + 1,
                col=1,
            )
        fig.update_layout(height=300 * tic_entry.planet_count)
        fig.update(layout_coloraxis_showscale=False)
        fname = os.path.join(tic_entry.outdir, FOLDED_LIGHTCURVE_PLOT)
        logger.debug(f"Saving {fname}")
        fig.write_image(fname)
        return fig
