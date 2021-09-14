import logging
import os
from typing import Optional, List

import matplotlib.pyplot as plt
import numpy as np
from pymc3.sampling import MultiTrace

from tess_atlas.data import TICEntry
from .labels import (
    LIGHTCURVE_PLOT,
    TIME_LABEL,
    FLUX_LABEL,
    FOLDED_LIGHTCURVE_PLOT,
    TIME_SINCE_TRANSIT_LABEL,
    PHASE_PLOT,
)
from .plotter_backend import PlotterBackend
from .plotting_utils import get_colors


class MatplotlibPlotter(PlotterBackend):
    @staticmethod
    def plot_lightcurve(
        tic_entry: TICEntry, model_lightcurves: Optional[List[float]] = None
    ) -> plt.Figure:
        """Plot lightcurve data + transit fits (if provided) in one plot"""
        if model_lightcurves is None:
            model_lightcurves = []
        colors = get_colors(len(model_lightcurves))
        fig, ax = plt.subplots(1, figsize=(7, 5))

        lc = tic_entry.lightcurve
        ax.scatter(
            lc.time, lc.flux, color="k", label="Data", s=0.75, alpha=0.5
        )
        ax.set_ylabel(FLUX_LABEL)
        ax.set_xlabel(TIME_LABEL)

        for i, model_lightcurve in enumerate(model_lightcurves):
            ax.plot(
                lc.time, model_lightcurve, label=f"Planet {i} fit", c=colors[i]
            )

        ax.legend(markerscale=5)

        fname = os.path.join(tic_entry.outdir, LIGHTCURVE_PLOT)
        logging.debug(f"Saving {fname}")
        plt.tight_layout()
        fig.savefig(fname)

    @staticmethod
    def plot_folded_lightcurve(
        tic_entry: TICEntry, model_lightcurves: Optional[List[float]] = None
    ) -> plt.Figure:
        """Subplots of folded lightcurves + transit fits (if provided) for each transit"""
        if model_lightcurves is None:
            model_lightcurves = []

        fig, axes = plt.subplots(
            tic_entry.planet_count, figsize=(7, 5 * tic_entry.planet_count)
        )
        if tic_entry.planet_count == 1:
            axes = [axes]
        colors = get_colors(tic_entry.planet_count)

        subplot_titles = [
            f"Planet {i + 1}: TOI-{c.toi_id}"
            for i, c in enumerate(tic_entry.candidates)
        ]

        for i in range(tic_entry.planet_count):
            lc = tic_entry.lightcurve
            planet = tic_entry.candidates[i]
            axes_cb = axes[i].scatter(
                planet.get_timefold(lc.time),
                lc.flux,
                c=lc.time,
                label=f"Data",
                s=0.75,
                alpha=0.25,
            )
            fig.colorbar(axes_cb, ax=axes[i], label=TIME_LABEL)

        for i, model_lightcurve in enumerate(model_lightcurves):
            lc = tic_entry.lightcurve
            planet = tic_entry.candidates[i]
            axes[i].scatter(
                planet.get_timefold(lc.time),
                model_lightcurve,
                label=f"Planet {i + 1} fit",
                s=5,
                c=colors[i],
            )

        for i, ax in enumerate(axes):
            ax.set_xlabel(TIME_SINCE_TRANSIT_LABEL)
            ax.set_ylabel(FLUX_LABEL)
            ax.set_title(subplot_titles[i])
            ax.legend(markerscale=5)

        plt.tight_layout()
        fname = os.path.join(tic_entry.outdir, FOLDED_LIGHTCURVE_PLOT)
        logging.debug(f"Saving {fname}")
        fig.savefig(fname)

    @staticmethod
    def plot_phase(tic_entry: TICEntry, trace: MultiTrace):
        colors = get_colors(tic_entry.planet_count)
        for i in range(tic_entry.planet_count):
            plt.figure(figsize=(7, 5))
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
                alpha=0.33,
            )

            inds = np.argsort(x_fold)
            inds = inds[np.abs(x_fold)[inds] < 0.3]
            pred = (
                trace["lightcurves"][:, inds, i] * 1e3 + trace["f0"][:, None]
            )
            pred = np.percentile(pred, [16, 50, 84], axis=0)
            plt.plot(x_fold[inds], pred[1], color=colors[i], label="model")
            art = plt.fill_between(
                x_fold[inds],
                pred[0],
                pred[2],
                color=colors[i],
                alpha=0.5,
                zorder=1000,
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
            plt.xlabel(TIME_SINCE_TRANSIT_LABEL)
            plt.ylabel(FLUX_LABEL)
            plt.title(f"Planet {i+1}")
            plt.xlim(-0.3, 0.3)
            plt.tight_layout()
            fname = os.path.join(
                tic_entry.outdir, PHASE_PLOT.replace(".", f"_{i+1}.")
            )
            logging.debug(f"Saving {fname}")
            plt.savefig(fname)
