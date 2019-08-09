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

    bp_mag = r["phot_bp_mean_mag"]
    bp_mag_err = r["phot_bp_mean_flux_error"] / r["phot_bp_mean_flux"]
    rp_mag = r["phot_rp_mean_mag"]
    rp_mag_err = r["phot_rp_mean_flux_error"] / r["phot_rp_mean_flux"]

    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    fn = os.path.join(output_dir, "{0}.h5".format(kicid))
    if os.path.exists(fn):
        return

    # Set up an isochrones model using the MIST tracks
    mist = isochrones.get_ichrone("mist", bands=["BP", "RP"])
    params = {
        "BP": (bp_mag, np.clip(bp_mag_err, 0.01, np.inf)),
        "RP": (rp_mag, np.clip(rp_mag_err, 0.01, np.inf)),
        "parallax": (plx, plx_err),
    }
    mod = isochrones.SingleStarModel(mist, max_distance=2000 / plx, **params)

    # These functions wrap isochrones so that they can be used with dynesty:
    def prior_transform(u):
        cube = np.copy(u)
        mod.mnest_prior(cube, None, None)
        return cube

    def loglike(theta):
        lp = mod.lnpost(theta)
        if np.isfinite(lp):
            return np.clip(lp, -1e10, np.inf)
        return -1e10

    # Run nested sampling on this model
    sampler = dynesty.NestedSampler(loglike, prior_transform, mod.n_params)
    strt = time.time()
    sampler.run_nested()
    print("Sampling took {0} minutes".format((time.time() - strt) / 60.0))

    # Resample the chain to get unit weight samples and update the isochrones
    # model
    results = sampler.results
    samples = dynesty.utils.resample_equal(
        results.samples, np.exp(results.logwt - results.logz[-1])
    )
    df = mod._samples = pd.DataFrame(dict(zip(mod.param_names, samples.T)))
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
