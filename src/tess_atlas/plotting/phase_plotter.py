import logging
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from arviz import InferenceData
from plotly.subplots import make_subplots

from ..data.inference_data_tools import get_samples_dataframe
from ..utils import NOTEBOOK_LOGGER_NAME
from .extra_plotting.ci import plot_ci, plot_xy_binned
from .labels import (
    FLUX_LABEL,
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
)

logger = logging.getLogger(NOTEBOOK_LOGGER_NAME)


def _preprocess_phase_plot_data(
    tic_entry, model, inference_data, initial_params, kwgs
):
    if initial_params:
        kwgs.update(
            dict(
                initial_params=initial_params,
                initial_lightcurves=generate_model_lightcurve(
                    model, initial_params
                ),
            )
        )

    if inference_data:
        # get posterior df + compute model vars
        lcs, gp_model, model_samples = get_lc_and_gp_from_inference_object(
            model, inference_data, n=kwgs.get("num_lc", 12)
        )
        kwgs.update(
            dict(
                inference_data=inference_data,
                posterior=get_samples_dataframe(inference_data),
                lcs=lcs,
                gp_model=gp_model,
                model_samples=model_samples,
            )
        )

    return kwgs


def plot_phase(
    tic_entry,
    model,
    inference_data: Optional[InferenceData] = None,
    initial_params: Optional[Dict] = None,
    **kwgs,
):
    """Adapted from exoplanet tutorials
    https://gallery.exoplanet.codes/tutorials/transit/#phase-plots

    In addition to {tic_entry, model} this function needs either
    - inference_data
    - initial_params
    """

    kwgs = _preprocess_phase_plot_data(
        tic_entry, model, inference_data, initial_params, kwgs
    )

    if kwgs.get("thumbnail", False):
        kwgs.update(
            data_bins=80,
            default_fs=0,
            period_fs=0,
            legend_fs=0,
            ms=2.5,
            savekwgs=dict(
                transparent=True, dpi=80, bbox_inches="tight", pad_inches=0
            ),
            annotate_with_period=False,
            figsize=(2.0, 1.5),
            legend=0,
        )
    figsize = kwgs.get("figsize", (7, 5))

    for i in range(tic_entry.planet_count):
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
        add_phase_data_to_ax(ax, i, tic_entry, **kwgs)


def _get_period_txt(p_std, p_mean):
    # generate period txt
    unc = ""
    if p_std:
        unc = "" if p_std < 0.001 else f" \\pm {p_std:.2f}"
    return f"$P = {p_mean:.2f}{unc}$ [days]"


def add_phase_data_to_ax(
    ax,
    i,
    tic_entry,
    gp_model=None,
    posterior=None,
    lcs=[],  # precomputed lcs
    initial_lightcurves=[],  # precomputed inital lcs
    model_samples=[],  # precomputed inital lcs
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
    default_fs: Optional[int] = 16,
    period_fs: Optional[int] = 12,
    legend_fs: Optional[int] = 10,
    ms: Optional[int] = 6,
    annotate_with_period: Optional[bool] = True,
    savekwgs=dict(transparent=False, dpi=150),
    save=True,
    legend=1,
    **kwgs,
):
    plt_min, plt_max = -0.3, 0.3
    lc = tic_entry.lightcurve
    t = lc.time
    yerr = lc.flux_err
    toi = tic_entry.toi_number
    colors = get_colors(tic_entry.planet_count)

    # ATM this is jsut the raw lc
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
                ith_flux -= np.median(initial_lightcurves[..., j], axis=(0, 1))

    # Plot the folded datapoints
    yerr = yerr if plot_error_bars else np.zeros(len(yerr))
    x_fold, idx = fold_data(lc, t0, p, plt_min, plt_max)
    xy_dat = (
        x_fold[idx],
        ith_flux[idx],
    )
    if plot_all_datapoints:
        axes_cb = ax.scatter(
            *xy_dat,
            c=t[idx],
            cmap="Greys",
            label=f"data",
            s=0.75,
            alpha=0.75,
        )
        plt.colorbar(axes_cb, ax=ax, label=TIME_LABEL)
    elif plot_data_ci:
        plot_ci(
            *xy_dat,
            ax,
            label="data",
            zorder=-1000,
            alpha=0.4,
            bins=35,
        )
    else:  # default
        plot_xy_binned(*xy_dat, yerr=yerr[idx], ax=ax, bins=data_bins, ms=ms)

    # Plot initial folded lightcurve model if present
    initial_line = None
    if len(initial_lightcurves) > 0:
        init_x, init_y = fold_lightcurve_models(
            lc,
            initial_lightcurves[..., i],
            [initial_params["tmin"][i]],
            [initial_params["p"][i]],
            plt_min,
            plt_max,
        )
        (initial_line,) = ax.plot(
            init_x,
            init_y[0],
            color=colors[i],
            label="initial fit",
            ls="dashed",
        )
        ylim = np.abs(np.min(np.hstack(init_y))) * 1.2

    # Plot lc bands if inference data present
    model_line = None
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
        (model_line,) = ax.plot(
            pred_x, quants[1, :], color=colors[i], label="model"
        )
        art = ax.fill_between(
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
    if annotate_with_period:
        ann = ax.annotate(
            _get_period_txt(p_std, p_mean),
            (0, 0),
            xycoords="axes fraction",
            xytext=(5, 5),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=period_fs,
        )

    # ax range
    if max(x_fold[idx]) < plt_max:
        plt_min, plt_max = min(x_fold[idx]), max(x_fold[idx])
    ax.set_xlim(plt_min, plt_max)
    if zoom_y_axis:
        ax.set_ylim(-ylim, ylim)

    # ax labels
    ax.set_xlabel(TIME_SINCE_TRANSIT_LABEL, fontsize=default_fs)
    ax.set_ylabel(FLUX_LABEL, fontsize=default_fs)

    if legend == 1:
        ax.legend(fontsize=legend_fs, loc="lower right")
    elif legend == 2:
        title = f"TOI {toi}.0{i + 1}\n{_get_period_txt(p_std, p_mean)}"
        handles, labels = [], []
        if initial_line:
            handles.append(initial_line)
            labels.append("ExoFOP")
        if model_line:
            handles.append(model_line)
            labels.append("Atlas")
        l = ax.legend(
            handles,
            labels,
            fontsize=legend_fs,
            loc="lower right",
            title=title,
            title_fontsize=legend_fs,
            frameon=True,
        )
        l._legend_box.align = "left"
        l.get_frame().set_linewidth(0.0)

    if not save:
        return

    if thumbnail:
        plt.axis("off")
        plt.margins(x=0, y=0, tight=True)
        plt.xlabel("")
        plt.ylabel("")
        plt.axis("tight")
    else:
        plt.title(f"TOI {toi}: Planet {i + 1}", fontsize=default_fs)

    plt.tight_layout()

    fname = PHASE_PLOT.replace(".", f"_TOI{toi}_{i + 1}.")
    if plot_label:
        fname = f"{plot_label}_{fname}"
    if thumbnail:
        fname = fname.replace(".png", "_thumbnail.png")

    fname = os.path.join(tic_entry.outdir, fname)
    logger.debug(f"Saving {fname}")
    plt.savefig(fname, **savekwgs)
    if thumbnail:
        plt.close()


def plot_folded_lightcurve(
    tic_entry: "TICEntry", model_lightcurves: Optional[List[float]] = None
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
