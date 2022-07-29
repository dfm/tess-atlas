import logging
import os


from typing import Optional, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from arviz import InferenceData
import plotly.graph_objects as go
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
    get_colors,
    get_lc_and_gp_from_inference_object,
    get_longest_unbroken_section_of_data,
    fold_data,
    generate_model_lightcurve,
    fold_lightcurve_models,
)

logger = logging.getLogger(NOTEBOOK_LOGGER_NAME)


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
    thumbnail: Optional[bool] = False,
):
    """Adapted from exoplanet tutorials
    https://gallery.exoplanet.codes/tutorials/transit/#phase-plots

    In addition to {tic_entry, model} this function needs either
    - inference_data
    - initial_params
    """

    if thumbnail:
        figsize = (2.0, 1.5)
        data_bins = 80
        default_fs, period_fs, legend_fs = 0, 0, 0
        ms = 2.5
        savekwg = dict(
            transparent=True, dpi=80, bbox_inches="tight", pad_inches=0
        )
    else:
        figsize = (7, 5)
        default_fs, period_fs, legend_fs = 16, 12, 10
        ms = 6
        savekwg = dict(transparent=False, dpi=150)

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
                alpha=0.75,
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
                *xy_dat, yerr=yerr[idx], ax=plt.gca(), bins=data_bins, ms=ms
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
                color=colors[i],
                label="initial fit",
                ls="dashed",
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
        ann = plt.annotate(
            txt,
            (0, 0),
            xycoords="axes fraction",
            xytext=(5, 5),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=period_fs,
        )

        if thumbnail:
            plt.axis("off")
            ann.remove()
            plt.margins(x=0, y=0, tight=True)
            plt.axis("tight")
        else:
            plt.legend(fontsize=legend_fs, loc="lower right")
            plt.xlabel(TIME_SINCE_TRANSIT_LABEL, fontsize=default_fs)
            plt.ylabel(FLUX_LABEL, fontsize=default_fs)
            plt.title(f"TOI {toi}: Planet {i + 1}", fontsize=default_fs)

        if max(x_fold[idx]) < plt_max:
            plt_min, plt_max = min(x_fold[idx]), max(x_fold[idx])

        plt.xlim(plt_min, plt_max)
        if zoom_y_axis:
            plt.ylim(-ylim, ylim)
        plt.tight_layout()

        fname = PHASE_PLOT.replace(".", f"_TOI{toi}_{i + 1}.")
        if plot_label:
            fname = f"{plot_label}_{fname}"
        if thumbnail:
            fname = fname.replace(".png", "_thumbnail.png")

        fname = os.path.join(tic_entry.outdir, fname)
        logger.debug(f"Saving {fname}")
        plt.savefig(fname, **savekwg)
        if thumbnail:
            plt.close()


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
                x=lc.timefold(t0=planet.tmin, p=planet.period),
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
                x=lc.timefold(t0=planet.tmin, p=planet.period),
                y=model_lightcurve,
                mode="markers",
                name=f"Planet {i + 1}",
            ),
            row=i + 1,
            col=1,
        )
    fig.update_layout(height=300 * tic_entry.planet_count)
    fig.update(layout_coloraxis_showscale=False)
    fname = os.path.join(tic_entry.outdir, PHASE_PLOT)
    logger.debug(f"Saving {fname}")
    fig.write_image(fname)
    return fig
