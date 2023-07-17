import os

import numpy as np
import pymc3_ext as pmx
from arviz import InferenceData

from tess_atlas.analysis import (
    calculate_eccentricity_weights,
    compute_variable,
    get_untransformed_varnames,
    sample_prior,
)
from tess_atlas.data import TICEntry
from tess_atlas.data.inference_data_tools import (
    get_optimized_init_params,
    summary,
    test_model,
)
from tess_atlas.logger import get_notebook_logger
from tess_atlas.plotting import (
    plot_diagnostics,
    plot_eccentricity_posteriors,
    plot_inference_trace,
    plot_lightcurve,
    plot_phase,
    plot_posteriors,
    plot_priors,
    plot_raw_lightcurve,
)


def test_sample_prior():
    pass
