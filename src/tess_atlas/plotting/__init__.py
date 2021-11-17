import os
from .corner_plotter import plot_posteriors, plot_eccentricity_posteriors
from .plotly_plots import PlotlyPlotter
from .matplotlib_plots import MatplotlibPlotter
from .histogram_plotter import plot_priors


if os.environ.get("INTERACTIVE_PLOTS", default="False") == "TRUE":
    plot_lightcurve = PlotlyPlotter.plot_lightcurve
    plot_folded_lightcurve = PlotlyPlotter.plot_folded_lightcurve
    plot_phase = MatplotlibPlotter.plot_phase
else:
    plot_lightcurve = MatplotlibPlotter.plot_lightcurve
    plot_folded_lightcurve = MatplotlibPlotter.plot_folded_lightcurve
    plot_phase = MatplotlibPlotter.plot_phase
