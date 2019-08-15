#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = ["run_fit"]

import os
import sys
import time
import numpy as np
import pandas as pd

from astropy.io import fits

import dynesty
import isochrones

with fits.open("kepler_dr2_1arcsec.fits") as f:
    xmatch = f[1].data

mist = isochrones.get_ichrone("mist", bands=["BP", "RP"])


def run_fit(kicid):
    matches = xmatch[xmatch["kepid"] == kicid]
    if not len(matches):
        print("skipping {0}".format(kicid))
        return

    ind = np.argmin(matches["kepler_gaia_ang_dist"])
    r = matches[ind]

    # Parallax offset reference: https://arxiv.org/abs/1805.03526
    plx = r["parallax"] + 0.082
    plx_err = np.sqrt(r["parallax_error"] ** 2 + 0.033 ** 2)
    if not (np.isfinite(plx) and np.isfinite(plx_err)):
        print("non finite parallax: {0}".format(kicid))
        return

    factor = 2.5 / np.log(10)
    params = {}
    for band in ["G", "BP", "RP"]:
        mag = float(r["phot_{0}_mean_mag".format(band.lower())])
        err = float(r["phot_{0}_mean_flux_error".format(band.lower())])
        err /= float(r["phot_{0}_mean_flux".format(band.lower())])
        err *= factor
        if not (np.isfinite(mag) and np.isfinite(err)):
            print("non finite params: {0}".format(kicid))
            return
        params[band] = (mag, err)
    jitter_vars = list(sorted(params.keys()))
    params["parallax"] = (float(plx), float(plx_err))
    for k, v in params.items():
        params[k] = np.array(v, dtype=np.float64)
    if params["parallax"][0] > 0:
        params["max_distance"] = np.clip(2000 / plx, 100, np.inf)
    print(params)

    output_dir = "astero_results"
    os.makedirs(output_dir, exist_ok=True)
    fn = os.path.join(output_dir, "{0}.h5".format(kicid))
    if os.path.exists(fn):
        return

    # Set up an isochrones model using the MIST tracks
    mist = isochrones.get_ichrone("mist", bands=["G", "BP", "RP"])
    mod = isochrones.SingleStarModel(mist, **params)

    # These functions wrap isochrones so that they can be used with dynesty:
    def prior_transform(u):
        cube = np.copy(u)
        mod.mnest_prior(cube[: mod.n_params], None, None)
        cube[mod.n_params :] = -10 + 20 * cube[mod.n_params :]
        return cube

    def loglike(theta):
        ind0 = mod.n_params
        lp0 = 0.0
        for i, k in enumerate(jitter_vars):
            err = np.sqrt(params[k][1] ** 2 + np.exp(theta[ind0 + i]))
            lp0 -= 2 * np.log(err)  # This is to fix a bug in isochrones
            mod.kwargs[k] = (params[k][0], err)
        lp = lp0 + mod.lnpost(theta[: mod.n_params])
        if np.isfinite(lp):
            return np.clip(lp, -1e10, np.inf)
        return -1e10

    # Run nested sampling on this model
    sampler = dynesty.NestedSampler(
        loglike, prior_transform, mod.n_params + len(jitter_vars)
    )
    strt = time.time()
    sampler.run_nested()
    print("Sampling took {0} minutes".format((time.time() - strt) / 60.0))

    # Resample the chain to get unit weight samples and update the isochrones
    # model
    results = sampler.results
    samples = dynesty.utils.resample_equal(
        results.samples, np.exp(results.logwt - results.logz[-1])
    )
    df = mod._samples = pd.DataFrame(
        dict(
            zip(
                list(mod.param_names) + ["log_jitter_" + k for k in jitter_vars],
                samples.T,
            )
        )
    )
    mod._derived_samples = mod.ic(*[df[c].values for c in mod.param_names])
    mod._derived_samples["parallax"] = 1000.0 / df["distance"]
    mod._derived_samples["distance"] = df["distance"]
    mod._derived_samples["AV"] = df["AV"]

    # Save these results to disk
    mod._samples.to_hdf(fn, "samples")
    mod._derived_samples.to_hdf(fn, "derived_samples")


if __name__ == "__main__":
    for kicid in sys.argv[1:]:
        print("Running KIC {0}".format(kicid))
        run_fit(int(kicid))
