import logging
import os

from tess_atlas.utils import NOTEBOOK_LOGGER_NAME

logger = logging.getLogger(NOTEBOOK_LOGGER_NAME)

from typing import Optional, Dict

import matplotlib.pyplot as plt
import numpy as np
from arviz import InferenceData

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
    fold_data,
    generate_model_lightcurve,
    fold_lightcurve_models,
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


def plot_phase(
    tic_entry,
    model,
    inference_data: Optional[InferenceData] = None,
    initial_params: Optional[Dict] = None,
    data_bins: Optional[int] = 200,
    plot_error_bars: Optional[bool] = False,
    plot_data_ci: Optional[bool] = False,
    plot_all_datapoints: Optional[bool] = False,
    zoom_y_axis: Optional[bool] = False,
    plot_label: Optional[str] = "",
    num_lc: Optional[int] = 12,
    low_res: Optional[bool] = False,
):
    """Adapted from exoplanet tutorials
    https://gallery.exoplanet.codes/tutorials/transit/#phase-plots

    In addition to {tic_entry, model} this function needs either
    - inference_data
    - initial_params
    """

    if low_res:
        figsize = (3.5, 2)
        dpi = 50
    else:
        figsize = (7, 5)
        dpi = 150

    # set some plotting constants
    plt_min, plt_max = -0.3, 0.3
    colors = get_colors(tic_entry.planet_count)
    lc = tic_entry.lightcurve
    t = lc.time
    yerr = lc.flux_err
    toi = tic_entry.toi_number

    if inference_data:
        # get posterior df + compute model vars
        posterior = get_samples_dataframe(inference_data)
        lcs, gp_model, model_samples = get_lc_and_gp_from_inference_object(
            model, inference_data, n=num_lc
        )

    initial_lightcurves = (
        []
        if initial_params is None
        else generate_model_lightcurve(model, initial_params)
    )

    for i in range(tic_entry.planet_count):
        plt.figure(figsize=figsize)

        ith_flux = lc.flux

        if inference_data:
            # get rid of noise in data
            ith_flux = ith_flux - gp_model

            # Get the posterior median orbital parameters
            pvals = posterior[f"p[{i}]"]
            p, p_mean, p_std = pvals.median(), pvals.mean(), pvals.std()
            t0 = np.median(posterior[f"tmin[{i}]"])

        else:  # if we only have the initial params
            p = initial_params["p"][i]
            p_mean, p_std = p, None
            t0 = initial_params["tmin"][i]

        # Compute the median of posterior estimate of the contribution from
        # the other planets and remove this from the data
        # (to plot just the planet we care about)
        for j in range(tic_entry.planet_count):
            if j != i:
                if inference_data:
                    ith_flux -= np.median(lcs[..., j], axis=(0, 1))
                else:
                    ith_flux -= np.median(
                        initial_lightcurves[..., j], axis=(0, 1)
                    )

        if plot_error_bars is False:
            yerr = np.zeros(len(yerr))

        # Plot the folded data
        x_fold, idx = fold_data(lc, t0, p, plt_min, plt_max)
        xy_dat = (
            x_fold[idx],
            ith_flux[idx],
        )
        if plot_all_datapoints:
            axes_cb = plt.scatter(
                *xy_dat,
                c=t[idx],
                cmap="Greys",
                label=f"data",
                s=0.75,
            )
            plt.colorbar(axes_cb, ax=plt.gca(), label=TIME_LABEL)
        elif plot_data_ci:
            plot_ci(
                *xy_dat,
                plt.gca(),
                label="data",
                zorder=-1000,
                alpha=0.4,
                bins=35,
            )
        else:  # default
            plot_xy_binned(
                *xy_dat,
                yerr=yerr[idx],
                ax=plt.gca(),
                bins=data_bins,
            )

        # Plot the folded lightcurve model
        if len(initial_lightcurves) > 0:
            init_x, init_y = fold_lightcurve_models(
                lc,
                initial_lightcurves[..., i],
                [initial_params["tmin"][i]],
                [initial_params["p"][i]],
                plt_min,
                plt_max,
            )
            plt.plot(
                init_x,
                init_y[0],
                color="tab:red",
                label="initial fit",
            )
            ylim = np.abs(np.min(np.hstack(init_y))) * 1.2

        if inference_data:
            pred_x, pred_y = fold_lightcurve_models(
                lc,
                lcs[..., i],
                t0s=[s["tmin"][i] for s in model_samples],
                periods=[s["p"][i] for s in model_samples],
                plt_min=plt_min,
                plt_max=plt_max,
            )
            quants = np.percentile(pred_y, [16, 50, 84], axis=0)
            plt.plot(pred_x, quants[1, :], color=colors[i], label="model")
            art = plt.fill_between(
                pred_x,
                quants[0, :],
                quants[2, :],
                color=colors[i],
                alpha=0.5,
                zorder=1000,
            )
            art.set_edgecolor("none")

            ylim = np.abs(np.min(np.hstack(pred_y))) * 1.2

        # Annotate the plot with the planet's period
        unc = f"+/- {p_std:.4f}" if p_std else ""
        txt = f"period = {p_mean:.4f} {unc} d"
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

        plt.legend(fontsize=10, loc="lower right")
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
        if low_res:
            fname = fname.replace(".png", "_lowres.png")

        fname = os.path.join(tic_entry.outdir, fname)
        logger.debug(f"Saving {fname}")
        plt.savefig(fname, dpi=dpi)
