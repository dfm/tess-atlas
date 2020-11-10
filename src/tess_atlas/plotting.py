# -*- coding: utf-8 -*-

__all__ = [
    "plot_lightcurve",
    "plot_folded_lightcurve",
    "plot_posteriors",
    "plot_eccentricity_posteriors",
    "get_range",
]

import logging
import os
from typing import List, Optional

import corner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pymc3 as pm
from plotly.subplots import make_subplots

from .data import TICEntry

# matplotlib settings
plt.style.use("default")
plt.rcParams["savefig.dpi"] = 100
plt.rcParams["figure.dpi"] = 100
plt.rcParams["font.size"] = 16
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Liberation Sans"]
plt.rcParams["font.cursive"] = ["Liberation Sans"]
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["image.cmap"] = "inferno"

CORNER_KWARGS = dict(
    smooth=0.9,
    label_kwargs=dict(fontsize=30),
    title_kwargs=dict(fontsize=16),
    color="#0072C1",
    truth_color="tab:orange",
    quantiles=[0.16, 0.84],
    levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9.0 / 2.0)),
    plot_density=False,
    plot_datapoints=False,
    fill_contours=True,
    max_n_ticks=3,
    verbose=False,
    use_math_text=True,
)

TIME_LABEL = "Time [days]"
FLUX_LABEL = "Relative Flux [ppt]"

LIGHTCURVE_PLOT = "flux_vs_time.png"
FOLDED_LIGHTCURVE_PLOT = "folded_flux_vs_time.png"
POSTERIOR_PLOT = "posteriors.png"
ECCENTRICITY_PLOT = "eccentricity_posteriors.png"


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
                x=lc.time, y=model_lightcurve, mode="lines", name=f"Planet {i}"
            )
        )
    fname = os.path.join(tic_entry.outdir, LIGHTCURVE_PLOT)
    logging.debug(f"Saving {fname}")
    fig.write_image(fname)
    return fig


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
    logging.debug(f"Saving {fname}")
    fig.write_image(fname)
    return fig


def plot_posteriors(
    tic_entry: TICEntry, trace: pm.sampling.MultiTrace
) -> None:
    samples = pm.trace_to_dataframe(trace, varnames=["p", "r", "b"])
    fig = corner.corner(samples, **CORNER_KWARGS, range=get_range(samples))
    fname = os.path.join(tic_entry.outdir, POSTERIOR_PLOT)
    logging.debug(f"Saving {fname}")
    fig.savefig(fname)


def plot_eccentricity_posteriors(
    tic_entry: TICEntry, ecc_samples: pd.DataFrame
) -> None:
    for n in range(tic_entry.planet_count):
        planet_n_samples = ecc_samples[[f"e[{n}]", f"omega[{n}]"]]
        fig = corner.corner(
            planet_n_samples,
            weights=ecc_samples[f"weights[{n}]"],
            labels=["eccentricity", "omega"],
            **CORNER_KWARGS,
            range=get_range(planet_n_samples),
        )
        plt.suptitle(f"Planet {n} Eccentricity")
        fname = os.path.join(
            tic_entry.outdir, f"planet_{n}_{ECCENTRICITY_PLOT}"
        )
        logging.debug(f"Saving {fname}")
        fig.savefig(fname)


def get_range(samples):
    return [[samples[l].min(), samples[l].max()] for l in samples]


def plot_phase(trace, tic_entry):

    for i in range(tic_entry.planet_count):
        plt.figure()
        p = np.median(trace["p"][:, i])
        t0 = np.median(trace["t0"][:, i])

        # Plot the folded data
        x_fold = (tic_entry.lightcurve.time - t0 + 0.5 * p) % p - 0.5 * p
        plt.errorbar(
            x_fold,
            tic_entry.lightcurve.flux,
            yerr=tic_entry.lightcurve.flux_err,
            fmt=".k",
            label="data",
            zorder=-1000,
        )

        inds = np.argsort(x_fold)
        inds = inds[np.abs(x_fold)[inds] < 0.3]
        pred = trace["lightcurves"][:, inds, i] * 1e3 + trace["f0"][:, None]
        pred = np.percentile(pred, [16, 50, 84], axis=0)
        plt.plot(x_fold[inds], pred[1], color="C1", label="model")
        art = plt.fill_between(
            x_fold[inds], pred[0], pred[2], color="C1", alpha=0.5, zorder=1000
        )
        art.set_edgecolor("none")

        # Annotate the plot with the planet's period
        txt = "period = {0:.4f} +/- {1:.4f} d".format(
            np.mean(trace["p"][:, i]), np.std(trace["p"][:, i])
        )
        plt.annotate(
            txt,
            (0, 0),
            xycoords="axes fraction",
            xytext=(5, 5),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=12,
        )

        plt.legend(fontsize=10, loc=4)
        plt.xlim(-0.5 * p, 0.5 * p)
        plt.xlabel("time since transit [days]")
        plt.ylabel("relative flux")
        plt.title("planet {0}".format(i + 1))
        plt.xlim(-0.3, 0.3)
