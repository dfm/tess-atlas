import logging
import os

from tess_atlas.utils import NOTEBOOK_LOGGER_NAME

logger = logging.getLogger(NOTEBOOK_LOGGER_NAME)

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from tess_atlas.data import TICEntry
from tess_atlas.data.inference_data_tools import get_samples_dataframe
from tess_atlas.utils import NOTEBOOK_LOGGER_NAME

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
from .plotting_utils import (
    get_colors,
    get_lc_and_gp_from_inference_object,
    get_longest_unbroken_section_of_data,
)

logger = logging.getLogger(NOTEBOOK_LOGGER_NAME)


class MatplotlibPlotter(PlotterBackend):
    @staticmethod
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

        ax.legend(markerscale=5, frameon=False)

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

    @staticmethod
    def plot_folded_lightcurve(
        tic_entry: TICEntry,
        model_lightcurves: Optional[np.ndarray] = None,
        save: Optional[bool] = True,
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
        if save:
            logger.debug(f"Saving {fname}")
            fig.savefig(fname)
        else:
            return fig

    @staticmethod
    def plot_phase(
        tic_entry,
        inference_data,
        model,
        initial_lightcurves: Optional[np.ndarray] = None,
        data_bins: Optional[int] = 200,
        plot_error_bars: Optional[bool] = False,
        plot_data_ci: Optional[bool] = False,
        plot_raw: Optional[bool] = False,
        zoom_y_axis: Optional[bool] = False,
        plot_label: Optional[str] = "",
        num_lc_for_phase: Optional[int] = 12,
    ):
        """Adapted from exoplanet tutorials
        https://gallery.exoplanet.codes/tutorials/transit/#phase-plots
        """

        # set some plotting constants
        plt_min, plt_max = -0.3, 0.3
        colors = get_colors(tic_entry.planet_count)
        t = tic_entry.lightcurve.time
        y = tic_entry.lightcurve.flux
        yerr = tic_entry.lightcurve.flux_err
        toi = tic_entry.toi_number

        # get posterior df + compute model vars
        posterior = get_samples_dataframe(inference_data)
        lcs, gp_model = get_lc_and_gp_from_inference_object(
            model, inference_data, n=num_lc_for_phase
        )
        # get rid of noise in data
        y = y - gp_model

        if initial_lightcurves is None:
            initial_lightcurves = []
        else:
            initial_lightcurves = initial_lightcurves.T

        for i in range(tic_entry.planet_count):
            plt.figure(figsize=(7, 5))

            # Get the posterior median orbital parameters
            pvals = posterior[f"p[{i}]"]
            p, p_mean, p_std = pvals.median(), pvals.mean(), pvals.std()
            t0 = np.median(posterior[f"tmin[{i}]"])

            # Compute the median of posterior estimate of the contribution from
            # the other planets and remove this from the data
            # (to plot just the planet we care about)
            ith_flux = y
            for j in range(tic_entry.planet_count):
                if j != i:
                    ith_flux -= np.median(lcs[..., j], axis=(0, 1))

            x_fold = (t - t0 + 0.5 * p) % p - 0.5 * p
            idx = np.where((plt_min <= x_fold) & (x_fold <= plt_max))

            if plot_error_bars is False:
                yerr = np.zeros(len(yerr))

            # Plot the folded data
            if plot_raw:
                plt.errorbar(
                    x_fold[idx],
                    ith_flux[idx],
                    yerr=yerr[idx],
                    fmt=".k",
                    label="data",
                    zorder=-1000,
                )
            elif plot_data_ci:
                plot_ci(
                    x_fold[idx],
                    ith_flux[idx],
                    plt.gca(),
                    label="data",
                    zorder=-1000,
                    alpha=0.4,
                    bins=35,
                )
            else:  # default
                plot_xy_binned(
                    x=x_fold[idx],
                    y=ith_flux[idx],
                    yerr=yerr[idx],
                    ax=plt.gca(),
                    bins=data_bins,
                )

            # Plot the folded lightcurve model
            inds = np.argsort(x_fold)
            inds = inds[np.abs(x_fold)[inds] < plt_max]

            if len(initial_lightcurves) > 0:
                lc = initial_lightcurves[i]
                plt.plot(
                    x_fold[inds],
                    lc[inds],
                    color="tab:red",
                    label="initial fit",
                )

            pred = lcs[..., inds, i]
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

            ylim = np.abs(np.min(pred)) * 1.2

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
            plt.title(f"TOI {toi}: Planet {i + 1}")
            plt.xlim(plt_min, plt_max)
            if zoom_y_axis:
                plt.ylim(-ylim, ylim)
            plt.tight_layout()

            fname = PHASE_PLOT.replace(".", f"_TOI{toi}_{i + 1}.")
            if plot_label:
                fname = f"{plot_label}_{fname}"

            fname = os.path.join(tic_entry.outdir, fname)
            logger.debug(f"Saving {fname}")
            plt.savefig(fname)
