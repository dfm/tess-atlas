# -*- coding: utf-8 -*-

__all__ = ["get_gaia_data", "fit_gaia_data"]

import os
import time

import dynesty
import isochrones
import numpy as np
import pandas as pd
from astroquery.gaia import Gaia

import astropy.units as u

from .config import OUTPUT_DIR
from .tess_atlas_version import __version__


def get_gaia_data(coord, approx_mag=None, radius=None, **kwargs):
    """Cross match to Gaia and construct a dataset for isochrone fitting

    Args:
        coord (SkyCoord): The coordinates of the source
        approx_mag (float, optional): The magnitude of the source in an
            optical band. If provided, only sources with a Gaia G mag within
            1 mag will be returned.
        radius (Quantity, optional): The angular search radius.

    Raises:
        ValueError: If no match is found or if any of the parameters are not
            finite in the catalog.

    Returns:
        The data dictionary

    """
    # coord = SkyCoord(ra=toi.RA, dec=toi.Dec, unit=(u.hourangle, u.deg))
    if radius is None:
        radius = 20 * u.arcsec
    j = Gaia.cone_search_async(coord, radius)
    r = j.get_results()

    # Only select targets within 1 mag of the target
    if approx_mag is not None:
        r = r[np.abs(r["phot_g_mean_mag"] - approx_mag) < 1]

    if not len(r):
        raise ValueError("no matches found")

    # Select the closest target
    r = r[0]

    # Parallax offset reference: https://arxiv.org/abs/1805.03526
    plx = r["parallax"] + 0.082
    plx_err = np.sqrt(r["parallax_error"] ** 2 + 0.033 ** 2)
    if not (np.isfinite(plx) and np.isfinite(plx_err)):
        raise ValueError("non finite parallax")

    # Convert flux error to magnitude error by linear propagation, this
    # should be ok for bright sources
    factor = 2.5 / np.log(10)
    params = {}
    for band in ["G", "BP", "RP"]:
        mag = float(r["phot_{0}_mean_mag".format(band.lower())])
        err = float(r["phot_{0}_mean_flux_error".format(band.lower())])
        err /= float(r["phot_{0}_mean_flux".format(band.lower())])
        err *= factor
        if not (np.isfinite(mag) and np.isfinite(err)):
            raise ValueError("non finite params")
        params[band] = (mag, err)

    params["parallax"] = (float(plx), float(plx_err))

    # Make sure that the dtypes are all correct
    for k, v in params.items():
        params[k] = np.array(v, dtype=np.float64)
    if params["parallax"][0] > 0:
        params["max_distance"] = np.clip(2000 / plx, 100, np.inf)

    params = dict(params, **kwargs)

    return params


def fit_gaia_data(name, gaia_data, clobber=False):
    # We will fit for jitter parameters for each magnitude
    jitter_vars = ["G", "BP", "RP"]

    # Set up an isochrones model using the MIST tracks
    mist = isochrones.get_ichrone("mist", bands=["G", "BP", "RP"])
    mod = isochrones.SingleStarModel(mist, **gaia_data)

    # Return the existing samples if not clobbering
    output_dir = os.path.join(OUTPUT_DIR, __version__, name)
    os.makedirs(output_dir, exist_ok=True)
    fn = os.path.join(output_dir, "star.h5")
    if (not clobber) and os.path.exists(fn):
        mod._samples = pd.read_hdf(fn, "samples")
        mod._derived_samples = pd.read_hdf(fn, "derived_samples")
        return mod

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
            err = np.sqrt(gaia_data[k][1] ** 2 + np.exp(theta[ind0 + i]))
            lp0 -= 2 * np.log(err)  # This is to fix a bug in isochrones
            mod.kwargs[k] = (gaia_data[k][0], err)
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
                list(mod.param_names)
                + ["log_jitter_" + k for k in jitter_vars],
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

    return mod
