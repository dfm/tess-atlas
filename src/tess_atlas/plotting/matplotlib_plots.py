import logging
import os
from typing import Optional

from tess_atlas.utils import NOTEBOOK_LOGGER_NAME

logger = logging.getLogger(NOTEBOOK_LOGGER_NAME)


from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from tess_atlas.data import TICEntry
from tess_atlas.data.inference_data_tools import (
    convert_to_samples_dict,
    get_posterior_samples,
)
from tess_atlas.utils import NOTEBOOK_LOGGER_NAME

from ..analysis import compute_variable, get_untransformed_varnames
from .extra_plotting.ci import plot_ci, plot_xy_binned
from .labels import (
    FLUX_LABEL,
    FOLDED_LIGHTCURVE_PLOT,
    LIGHTCURVE_PLOT,
    PHASE_PLOT,
    TIME_LABEL,
    TIME_SINCE_TRANSIT_LABEL,
)
from .plotter_backend import PlotterBackend
from .plotting_utils import get_colors

logger = logging.getLogger(NOTEBOOK_LOGGER_NAME)


class MatplotlibPlotter(PlotterBackend):
    @staticmethod
    def plot_lightcurve(
        tic_entry: TICEntry, model_lightcurves: Optional[np.ndarray] = None
    ) -> plt.Figure:
        """Plot lightcurve data + transit fits (if provided) in one plot

        model_lightcurves: (lightcurve_num, lightcurve_y_vals, planet_id)

        """
        if model_lightcurves is None:
            model_lightcurves = []
        else:
            model_lightcurves = model_lightcurves.T

        colors = get_colors(tic_entry.planet_count)
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
        logger.debug(f"Saving {fname}")
        plt.tight_layout()
        fig.savefig(fname)

    @staticmethod
    def plot_folded_lightcurve(
        tic_entry: TICEntry, model_lightcurves: Optional[np.ndarray] = None
    ) -> plt.Figure:
        """Subplots of folded lightcurves + transit fits (if provided) for each transit"""
        if model_lightcurves is None:
            model_lightcurves = []
        else:
            model_lightcurves = model_lightcurves.T

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
        logger.debug(f"Saving {fname}")
        fig.savefig(fname)

    @staticmethod
    def plot_phase(
        tic_entry,
        inference_data,
        model,
        plot_data_ci: Optional[bool] = False,
        plot_binned: Optional[bool] = False,
    ):
        """Adapted from exoplanet tutorials
        https://gallery.exoplanet.codes/tutorials/transit/#phase-plots
        """
        colors = get_colors(tic_entry.planet_count)

        t = tic_entry.lightcurve.time
        y = tic_entry.lightcurve.flux
        yerr = tic_entry.lightcurve.flux_err

        plt_min, plt_max = -0.3, 0.3

        for i in range(tic_entry.planet_count):
            plt.figure(figsize=(7, 5))

            varnames = get_untransformed_varnames(model)
            samples = get_posterior_samples(
                inference_data=inference_data, varnames=varnames, size=1000
            )

            lcs = compute_variable(
                model=model, samples=samples, target=model.lightcurve_models
            )

            samples_dict = convert_to_samples_dict(varnames, samples)

            # Get the posterior median orbital parameters
            p = np.median(samples_dict["p"])
            p_mean = np.mean(samples_dict["p"])
            p_std = np.std(samples_dict["p"])
            t0 = np.median(samples_dict["t0"])
            f0_array = samples_dict["f0"]

            # Compute the median of posterior estimate of the contribution from
            # the other planets and remove this from the data
            # (to plot just the planet we care about)
            ith_flux = y
            for j in range(tic_entry.planet_count):
                if j != i:
                    ith_flux -= np.median(lcs[..., j], axis=(0, 1))
            # TODO: check if this works for multi-planet systems

            x_fold = (t - t0 + 0.5 * p) % p - 0.5 * p
            idx = np.where((plt_min <= x_fold) & (x_fold <= plt_max))

            # Plot the folded data
            if not plot_data_ci:
                plt.errorbar(
                    x_fold,
                    ith_flux,
                    yerr=yerr,
                    fmt=".k",
                    label="data",
                    zorder=-1000,
                )
            else:
                plot_ci(
                    x_fold[idx],
                    ith_flux[idx],
                    plt.gca(),
                    label="data",
                    zorder=-1000,
                    alpha=0.4,
                    bins=35,
                )

            if plot_binned:
                plot_xy_binned(
                    x_fold[idx],
                    ith_flux[idx],
                    plt.gca(),
                    bins=500,
                    color="gray",
                )

            # Plot the folded model
            inds = np.argsort(x_fold)
            inds = inds[np.abs(x_fold)[inds] < plt_max]
            scaled_lcs = lcs[..., inds, i] * 1e3

            pred = scaled_lcs.copy()
            for pred_num, f0 in enumerate(f0_array):
                pred[pred_num, :] -= f0

            quants = np.percentile(pred, [16, 50, 84], axis=0)
            plt.plot(
                x_fold[inds], quants[1, :], color=colors[i], label="model"
            )
            art = plt.fill_between(
                x_fold[inds],
                quants[0, :],
                quants[2, :],
                color=colors[i],
                alpha=0.5,
                zorder=1000,
            )
            art.set_edgecolor("none")

            # Annotate the plot with the planet's period

            txt = f"period = {p_mean:.4f} +/- {p_std:.4f} d"
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
            plt.title(f"Planet {i + 1}")
            plt.xlim(plt_min, plt_max)
            plt.tight_layout()
            fname = os.path.join(
                tic_entry.outdir, PHASE_PLOT.replace(".", f"_{i + 1}.")
            )
            logger.debug(f"Saving {fname}")
            plt.savefig(fname)
