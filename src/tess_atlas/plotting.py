# -*- coding: utf-8 -*-

__all__ = [
    "plot_lightcurve_and_masks",
    "plot_masked_lightcurve_flux_vs_time_since_transit",
    "plot_lightcurve_with_inital_model",
]

from typing import List, Optional

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .data import TICEntry


def plot_lightcurve_and_masks(tic_entry: TICEntry):
    lc = tic_entry.lightcurve
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        x_title="Time [days]",
    )
    fig.add_trace(
        go.Scattergl(
            x=lc.time,
            y=lc.flux,
            mode="lines+markers",
            marker_color="black",
            marker_size=2,
            line_width=0.1,
            hoverinfo="skip",
            name="Data",
        ),
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="Relative Flux [ppt]", row=1, col=1)
    fig.add_trace(
        go.Scattergl(
            x=lc.time,
            y=tic_entry.get_combined_mask(),
            mode="lines",
            line=dict(color="white"),
            fill="tozeroy",
            name=f"Combined",
        ),
        row=2,
        col=1,
    )
    for i, candidate in enumerate(tic_entry.candidates):
        fig.add_trace(
            go.Scattergl(
                x=lc.time,
                y=candidate.get_mask(lc.time),
                mode="lines",
                name=f"Planet {i+1}",
            ),
            row=2,
            col=1,
        )

    fig.update_yaxes(title_text="Planet Transiting", row=2, col=1)
    return fig


def plot_masked_lightcurve_flux_vs_time_since_transit(
    tic_entry: TICEntry, model_lightcurves: Optional[List[float]] = None
):
    if model_lightcurves is None:
        model_lightcurves = []
    num_planets = tic_entry.planet_count
    subplot_titles = [
        f"Planet {i+1}: TOI-{c.toi_id}"
        for i, c in enumerate(tic_entry.candidates)
    ]
    fig = make_subplots(
        rows=num_planets,
        cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.1,
    )
    for i in range(num_planets):
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
                name=f"Candidate {i+1} Data",
            ),
            row=i + 1,
            col=1,
        )
        fig.update_xaxes(title_text="Time [days]", row=i + 1, col=1)
        fig.update_yaxes(title_text="Relative Flux [ppt]", row=i + 1, col=1)
    for i, model_lightcurve in enumerate(model_lightcurves):
        lc = tic_entry.lightcurve
        planet = tic_entry.candidates[i]
        fig.add_trace(
            go.Scattergl(
                x=planet.get_timefold(lc.time),
                y=model_lightcurve,
                mode="markers",
                name=f"Planet {i+1}",
            ),
            row=i + 1,
            col=1,
        )
    fig.update_layout(height=300 * num_planets)
    fig.update(layout_coloraxis_showscale=False)
    return fig


def plot_lightcurve_with_inital_model(tic_entry: TICEntry, map_soln):
    lc = tic_entry.lightcurve
    fig = go.Figure()
    make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    fig.add_trace(
        go.Scattergl(
            x=lc.time,
            y=lc.flux,
            mode="lines+markers",
            marker_color="black",
            marker_size=2,
            line_width=0.1,
            hoverinfo="skip",
            name="Data",
        )
    )
    for i in range(tic_entry.planet_count):
        fig.add_trace(
            go.Scattergl(
                x=lc.time,
                y=map_soln["lightcurves"][:, i] * 1e3,
                mode="lines",
                name=f"Planet {i}",
            )
        )
    fig.update_layout(
        xaxis_title="Time [days]", yaxis_title="Relative Flux [ppt]"
    )
    return fig
