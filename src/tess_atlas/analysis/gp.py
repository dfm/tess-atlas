# -*- coding: utf-8 -*-

__all__ = ["get_kernel_params", "setup_kernel"]

from typing import Optional

import exoplanet as xo
import numpy as np
import pymc3 as pm
import pymc3_ext as pmx
from celerite2.theano import GaussianProcess, terms

from tess_atlas.data import TICEntry


def get_kernel_params(tic_entry: TICEntry):
    # Get the light curve without transits
    mask = np.array(tic_entry.get_combined_mask(), dtype=bool)
    lc = tic_entry.lightcurve
    x = lc.time[~mask]
    y = lc.flux[~mask]
    yerr = lc.flux_err[~mask]

    # Guess the period
    results = xo.estimators.autocorr_estimator(x, y, smooth=200)
    period_guess = results["peaks"][0]["period"]

    std = np.std(y)
    with pm.Model() as test_model:
        mean = pm.Normal("mean", mu=0.0, sigma=1000.0)
        sigma0 = pm.Lognormal("sigma0", mu=np.log(std), sigma=10.0)
        sigma1 = pm.Lognormal("sigma1", mu=np.log(std), sigma=10.0)
        rho = pm.Lognormal("rho", mu=np.log(10 * period_guess), sigma=10.0)
        sigma2 = pm.Lognormal("sigma2", mu=np.log(std), sigma=10.0)
        period = pm.Lognormal("period", mu=np.log(period_guess), sigma=10.0)
        Q = pm.Lognormal("Q", mu=0.0, sigma=10.0)
        dQ = pm.Lognormal("dQ", mu=0.0, sigma=10.0)
        f = pmx.UnitUniform("f")

        kernel = terms.SHOTerm(sigma=sigma1, rho=rho, Q=0.3)
        kernel += terms.RotationTerm(
            sigma=sigma2, period=period, Q0=Q, dQ=dQ, f=f
        )

        gp = GaussianProcess(
            kernel, t=x, diag=yerr ** 2 + sigma0 ** 2, mean=mean, quiet=True
        )
        gp.marginal("obs", observed=y)

        kwargs = dict(verbose=False)
        best = (np.inf, None)
        for factor in [0.5, 1.0, 2.0]:
            map_soln = test_model.test_point
            map_soln["period_log__"] = np.log(period_guess) + np.log(factor)
            map_soln = pmx.optimize(
                map_soln, [mean, sigma0, sigma1, sigma2], **kwargs
            )
            map_soln = pmx.optimize(
                map_soln, [mean, sigma0, sigma2, Q, dQ, f], **kwargs
            )
            map_soln = pmx.optimize(
                map_soln, [mean, sigma0, sigma1, rho], **kwargs
            )
            map_soln, info = pmx.optimize(map_soln, return_info=True, **kwargs)
            if info.fun < best[0]:
                best = (info.fun, map_soln)

    return best[1]


def setup_kernel(map_soln, model: Optional[pm.Model] = None):
    with pm.Model(name="gp", model=model):
        sigma0 = pm.Lognormal(
            "sigma0", mu=np.log(map_soln["sigma0"]), sigma=np.log(2)
        )
        sigma1 = pm.Lognormal(
            "sigma1", mu=np.log(map_soln["sigma1"]), sigma=np.log(2)
        )
        rho = pm.Lognormal("rho", mu=np.log(map_soln["rho"]), sigma=np.log(2))
        sigma2 = pm.Lognormal(
            "sigma2", mu=np.log(map_soln["sigma2"]), sigma=np.log(2)
        )
        period = pm.Lognormal(
            "period", mu=np.log(map_soln["period"]), sigma=np.log(1.1)
        )
        Q = pm.Lognormal("Q", mu=np.log(map_soln["Q"]), sigma=np.log(1.1))
        dQ = pm.Lognormal("dQ", mu=np.log(map_soln["dQ"]), sigma=np.log(1.1))
        f = pmx.UnitUniform("f", testval=map_soln["f"])

    kernel = terms.SHOTerm(sigma=sigma1, rho=rho, Q=0.3)
    kernel += terms.RotationTerm(sigma=sigma2, period=period, Q0=Q, dQ=dQ, f=f)

    return kernel, [sigma0, sigma1, rho, sigma2, period, Q, dQ, f]
