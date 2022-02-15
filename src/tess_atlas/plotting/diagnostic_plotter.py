import matplotlib.pyplot as plt
import numpy as np
import arviz as az
import os

from .matplotlib_plots import MatplotlibPlotter
from .plotting_utils import (
    get_longest_unbroken_section_of_data,
    get_colors,
    get_lc_and_gp_from_inference_object,
)
from .labels import DIAGNOSTIC_LIGHTCURVE_PLOT, DIAGNOSTIC_TRACE_PLOT


def plot_lightcurve_gp_and_residuals(tic_entry, model, zoom_in=True):
    "Adapted from https://gallery.exoplanet.codes/tutorials/tess/"

    colors = get_colors(tic_entry.planet_count)
    t = tic_entry.lightcurve.time
    y = tic_entry.lightcurve.flux
    lcs, gp_model = get_lc_and_gp_from_inference_object(
        model, tic_entry.inference_data
    )
    if zoom_in:
        idx, _ = get_longest_unbroken_section_of_data(t)
    else:
        idx = [i for i in range(len(t))]

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    ax = axes[0]
    ax.plot(t[idx], y[idx], "k", label="data")
    ax.plot(t[idx], gp_model[idx], color="gray", label="gp model")
    ax.legend(fontsize=10, loc=3)
    ax.set_ylabel("relative flux")

    ax = axes[1]
    ax.plot(t[idx], y[idx] - gp_model[idx], "k", label="de-trended data")
    net_lc = np.zeros(len(t))
    for i in range(tic_entry.planet_count):
        lc = np.median(lcs[..., i], axis=0)
        net_lc += lc
        snr = tic_entry.candidates[i].snr
        ax.plot(
            t[idx],
            lc[idx],
            label=f"Planet {i} (SNR {snr:.2f})",
            color=colors[i],
        )
    ax.legend(fontsize=10, loc=3)
    ax.set_ylabel("de-trended flux")

    ax = axes[2]
    models = gp_model + net_lc
    resid = y - models
    rms = np.sqrt(np.median(resid ** 2))
    mask = np.abs(resid) < 5 * rms
    total_outliers = np.sum(~mask)

    ax.plot(t[idx], resid[idx], "k", label=f"residuals")
    ax.plot(
        t[~mask],
        resid[~mask],
        "xr",
        label=f"outliers ({total_outliers} total)",
    )
    ax.axhline(0, color="#aaaaaa", lw=1, label="zero-line")
    ax.set_ylabel("residuals")
    ax.legend(fontsize=10, loc=3)
    ax.set_xlim(t[idx].min(), t[idx].max())
    ax.set_xlabel("time [days]")

    if zoom_in:
        perc_data = int(100 * (len(idx) / len(t)))
        fig.suptitle(f"{perc_data}% Data Displayed")
    fig.savefig(os.path.join(tic_entry.outdir, DIAGNOSTIC_LIGHTCURVE_PLOT))


def trace_plot(tic_entry):
    with az.style.context("default", after_reset=True):
        plt.close("all")
        az.plot_trace(
            tic_entry.inference_data,
            divergences="top",
            legend=True,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(os.path.join(tic_entry.outdir, DIAGNOSTIC_TRACE_PLOT))


def plot_diagnostics(tic_entry, model):
    plot_lightcurve_gp_and_residuals(tic_entry, model)
    trace_plot(tic_entry)
    MatplotlibPlotter.plot_phase(
        tic_entry,
        tic_entry.inference_data,
        model,
        plot_data_ci=True,
        plot_label="data_ci",
    )
    plt.close("all")
