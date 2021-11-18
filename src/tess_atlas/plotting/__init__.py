import os

from .corner_plotter import plot_eccentricity_posteriors, plot_posteriors
from .histogram_plotter import plot_priors
from .matplotlib_plots import MatplotlibPlotter
from .plotly_plots import PlotlyPlotter

if os.environ.get("INTERACTIVE_PLOTS", default="False") == "TRUE":
    plot_lightcurve = PlotlyPlotter.plot_lightcurve
    plot_folded_lightcurve = PlotlyPlotter.plot_folded_lightcurve
    plot_phase = MatplotlibPlotter.plot_phase
else:
    plot_lightcurve = MatplotlibPlotter.plot_lightcurve
    plot_folded_lightcurve = MatplotlibPlotter.plot_folded_lightcurve
    plot_phase = MatplotlibPlotter.plot_phase
