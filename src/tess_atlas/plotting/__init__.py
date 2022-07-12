import os

from .corner_plotter import plot_eccentricity_posteriors, plot_posteriors
from .histogram_plotter import plot_priors
from . import matplotlib_plots
from . import plotly_plots
from .diagnostic_plotter import (
    plot_diagnostics,
    plot_inference_trace,
    plot_raw_lightcurve,
)

if os.environ.get("INTERACTIVE_PLOTS", default="False") == "TRUE":
    plot_lightcurve = plotly_plots.plot_lightcurve
    plot_folded_lightcurve = plotly_plots.plot_folded_lightcurve
    plot_phase = matplotlib_plots.plot_phase
else:
    plot_lightcurve = matplotlib_plots.plot_lightcurve
    plot_folded_lightcurve = matplotlib_plots.plot_folded_lightcurve
    plot_phase = matplotlib_plots.plot_phase
