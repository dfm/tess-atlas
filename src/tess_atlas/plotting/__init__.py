import os
from .corner_plotter import plot_posteriors, plot_eccentricity_posteriors
from .plotly_plots import PlotlyPlotter
from .matplotlib_plots import MatplotlibPlotter

if os.environ.get("INTERACTIVE_PLOTS", default=False):
    plot_lightcurve = PlotlyPlotter.plot_lightcurve
    plot_folded_lightcurve = PlotlyPlotter.plot_folded_lightcurve
else:
    plot_lightcurve = MatplotlibPlotter.plot_lightcurve
    plot_folded_lightcurve = MatplotlibPlotter.plot_folded_lightcurve
