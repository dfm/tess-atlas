import logging
import os

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

from tess_atlas.utils import NOTEBOOK_LOGGER_NAME

logger = logging.getLogger(NOTEBOOK_LOGGER_NAME)

from ..data.data_utils import residual_rms
from .labels import (
    DIAGNOSTIC_LIGHTCURVE_PLOT,
    DIAGNOSTIC_RAW_LC_PLOT,
    DIAGNOSTIC_TRACE_PLOT,
)
from .phase_plotter import plot_phase
from .plotting_utils import (
    get_colors,
    get_lc_and_gp_from_inference_object,
    get_longest_unbroken_section_of_data,
)


def plot_raw_lightcurve(tic_entry, save=True, zoom_in=False):
    lc = tic_entry.lightcurve
    ax = lc.raw_lc.scatter(
        label=f"Raw Data ({len(lc.raw_lc):,} pts)",
        color="black",
    )
    lc.cleaned_lc.scatter(
        ax=ax,
        label=f"Cleaned Data ({len(lc.cleaned_lc):,} pts)",
        color="gray",
    )
    for i, p in enumerate(tic_entry.candidates):
        pi = f"[{i}]"
        t0, tmin, tmax, T = p.t0, p.tmin, p.tmax, p.period
        s = p.has_data_only_for_single_transit
        single = "Y" if s else "N"
        Np = p.num_periods
        t1 = t0 + T
        y = 1
        c = dict(color=f"C{i}")
        ca = dict(alpha=0.5, **c)
        yrng = dict(ymin=1 - p.depth * 1e-3, ymax=1)
        ax.scatter(
            [tmin + (i * T) for i in range(Np + 1)],
            [y] * (Np + 1),
            **c,
            alpha=1,
            marker="o",
            label=f"N{pi} transits: {Np+1} (single? {single})",
        )
        ax.plot(
            [t0, t1],
            [y, y],
            label=f"$T{pi}: {T:.2f}$ days",
            **ca,
            marker="s",
            mec="k",
            zorder=10,
        )
        ax.scatter(
            [t0, t1],
            [yrng["ymin"], yrng["ymin"]],
            **ca,
            marker="s",
            ec="k",
            zorder=10,
        )
        ax.vlines(
            [t0, t1],
            **yrng,
            label=f"$t_0{pi} - t_1{pi}: {t0:.2f}- {t1:.2f}$",
            **ca,
        )
        ax.vlines(
            [tmin, tmax],
            **yrng,
            ls="--",
            label="$t_{\\rm min}"
            f"{pi}"
            " - t_{\\rm max}"
            f"{pi}: {tmin:.2f}-{tmax:.2f}$",
            **ca,
            lw=2.5,
        )

    if zoom_in:
        idx, t = get_longest_unbroken_section_of_data(lc.time)
        perc_data = int(100 * (len(idx) / len(lc.time)))
        xrange = (min(t), max(t))
        if perc_data > 98:
            minx = lc.time[idx[0]]
            maxx = lc.time[idx[int(len(idx) / 2)]]
            xrange = (minx, maxx)
        ax.set_xlim(*xrange)
        txt = (
            f"{perc_data}% Data (full {int(min(lc.time))}-{int(max(lc.time))})"
        )
        ax.set_title(txt)

    l = plt.legend(
        loc="upper left",
        title=f"TOI {tic_entry.toi_number}",
        fontsize="x-small",
        frameon=False,
        bbox_to_anchor=(1.1, 1),
    )
    l._legend_box.align = "left"
    # plt.tight_layout()
    if save:
        plt.savefig(
            os.path.join(tic_entry.outdir, DIAGNOSTIC_RAW_LC_PLOT),
            bbox_inches="tight",
        )
    else:
        return ax.get_figure()


def plot_lightcurve_gp_and_residuals(
    tic_entry, model, zoom_in=True, num_lc=12
):
    "Adapted from https://gallery.exoplanet.codes/tutorials/tess/"
    # todo plot the maximum posterior param
    colors = get_colors(tic_entry.planet_count)
    t = tic_entry.lightcurve.time
    y = tic_entry.lightcurve.flux
    lcs, gp_model, _ = get_lc_and_gp_from_inference_object(
        model, tic_entry.inference_data, n=num_lc
    )
    raw_lc = tic_entry.lightcurve.raw_lc
    raw_t, raw_y = raw_lc.time.value, 1e3 * (raw_lc.flux.value - 1)

    if zoom_in:
        idx, _ = get_longest_unbroken_section_of_data(t)
    else:
        idx = [i for i in range(len(t))]

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    ax = axes[0]
    ax.scatter(raw_t, raw_y, c="gray", label="raw data", s=1, alpha=0.5)
    ax.scatter(t[idx], y[idx], c="k", label="data", s=1)
    net_lc = np.zeros(len(t))
    for i in range(tic_entry.planet_count):
        lc = np.median(lcs[..., i], axis=0)
        net_lc += lc
        snr = tic_entry.candidates[i].snr
        ax.plot(
            t[idx],
            lc[idx],
            label=f"Planet {i+1} (SNR {snr:.2f})",
            color=colors[i],
        )

    ax.legend(fontsize=10, loc=3)
    ax.set_ylabel("flux")

    ax = axes[1]
    ax.scatter(t[idx], y[idx] - net_lc[idx], c="k", label="data-lc", s=1)
    ax.plot(t[idx], gp_model[idx], color="gray", label="gp model")
    ax.legend(fontsize=10, loc=3)
    ax.set_ylabel("de-trended flux")

    ax = axes[2]
    models = gp_model + net_lc
    resid = y - models
    rms = residual_rms(resid)
    rms_mult = 5
    rms_threshold = rms * rms_mult
    mask = np.abs(resid) < rms_threshold
    total_outliers = np.sum(~mask)

    ax.scatter(t[idx], resid[idx], c="k", label=f"residuals", s=1)
    ax.plot(
        t[~mask],
        resid[~mask],
        "xr",
        label=f"outliers ({total_outliers})",
    )
    ax.axhline(
        -rms_threshold,
        color="red",
        ls="--",
        lw=1,
        label=f"rms * {rms_mult} (rms={rms:.4f})",
    )
    ax.axhline(rms_threshold, color="red", ls="--", lw=1)
    ax.axhline(0, color="#aaaaaa", lw=1, label="zero-line")
    ax.set_ylabel("residuals")
    ax.legend(fontsize=10, loc=3)
    ax.set_xlim(t[idx].min(), t[idx].max())
    ax.set_xlabel("time [days]")

    if total_outliers > 100:
        logger.warning(
            "Large number of outliers in residuals after fitting model."
        )

    if zoom_in:
        perc_data = int(100 * (len(idx) / len(t)))
        fig.suptitle(f"{perc_data}% Data Displayed")
    fig.subplots_adjust(hspace=0, wspace=0)
    fig.savefig(os.path.join(tic_entry.outdir, DIAGNOSTIC_LIGHTCURVE_PLOT))


def plot_inference_trace(tic_entry):
    with az.style.context("default", after_reset=True):
        az.plot_trace(
            tic_entry.inference_data,
            divergences="top",
            legend=True,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(os.path.join(tic_entry.outdir, DIAGNOSTIC_TRACE_PLOT))


def plot_diagnostics(tic_entry, model, init_params):
    plot_lightcurve_gp_and_residuals(tic_entry, model)
    plot_phase(
        tic_entry,
        model,
        tic_entry.inference_data,
        plot_data_ci=True,
        plot_label="data_ci",
    )
    plot_phase(
        tic_entry,
        model,
        tic_entry.inference_data,
        initial_params=init_params,
        thumbnail=True,
    )
