from typing import Dict, List, Optional

import arviz as az
import numpy as np
from pymc3 import Model

from tess_atlas.data.inference_data_tools import (
    get_median_sample,
    get_posterior_samples,
    get_samples_dataframe,
)

from ...analysis import compute_variable, get_untransformed_varnames


def get_lc_and_gp_from_inference_object(
    model: Model, inference_data: az.InferenceData, n: int = 12
):
    f0 = np.median(get_samples_dataframe(inference_data)[f"f0"])
    varnames = get_untransformed_varnames(model)
    samples = get_posterior_samples(
        inference_data=inference_data, varnames=varnames, size=n
    )
    median_sample = get_median_sample(
        inference_data=inference_data, varnames=varnames
    )
    lcs = compute_variable(model, samples, target=model.lightcurve_models)
    gp_mu = compute_variable(
        model, [median_sample], target=model.gp_mu, verbose=False
    )
    lcs = lcs * 1e3  # scale the lcs
    gp_model = gp_mu[0] + f0
    samples = [{k: v for k, v in zip(varnames, s)} for s in samples]
    return lcs, gp_model, samples


def generate_model_lightcurve(
    planet_transit_model,
    sample: Optional[Dict] = None,
    b: Optional[float] = None,
    dur: Optional[float] = None,
    f0: Optional[float] = None,
    jitter: Optional[float] = None,
    p: Optional[float] = None,
    r: Optional[float] = None,
    rho: Optional[float] = None,
    rho_circ: Optional[float] = None,
    sigma: Optional[float] = None,
    t0: Optional[float] = None,
    tmax: Optional[float] = None,
    u: Optional[List] = None,
):
    """Function to manually test some parameter values for the model"""
    if sample is None:
        assert b is not None, "Either pass 1 sample-dict or individual params"
        tmax_1 = tmax
        samp = [
            [b],
            [dur],
            f0,
            jitter,
            [p],
            [r],
            rho,
            [rho_circ],
            sigma,
            [t0],
            [tmax],
            tmax_1,
            u,
        ]
    else:
        model_varnames = get_untransformed_varnames(planet_transit_model)
        samp = [sample[n] for n in model_varnames]

    lc = compute_variable(
        model=planet_transit_model,
        samples=[samp],
        target=planet_transit_model.lightcurve_models,
    )
    return np.array(lc * 1e3)
