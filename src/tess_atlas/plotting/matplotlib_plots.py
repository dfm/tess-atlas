from typing import Optional, List

import matplotlib.pyplot as plt
import numpy as np

from tess_atlas.data import TICEntry
from .plotter_backend import PlotterBackend


class MatplotlibPlotter(PlotterBackend):
    @staticmethod
    def plot_lightcurve(
        tic_entry: TICEntry, model_lightcurves: Optional[List[float]] = None
    ):
        pass

    @staticmethod
    def plot_folded_lightcurve(
        tic_entry: TICEntry, model_lightcurves: Optional[List[float]] = None
    ):
        pass

    @staticmethod
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
            pred = (
                trace["lightcurves"][:, inds, i] * 1e3 + trace["f0"][:, None]
            )
            pred = np.percentile(pred, [16, 50, 84], axis=0)
            plt.plot(x_fold[inds], pred[1], color="C1", label="model")
            art = plt.fill_between(
                x_fold[inds],
                pred[0],
                pred[2],
                color="C1",
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
            plt.xlim(-0.5 * p, 0.5 * p)
            plt.xlabel("time since transit [days]")
            plt.ylabel("relative flux")
            plt.title("planet {0}".format(i + 1))
            plt.xlim(-0.3, 0.3)
