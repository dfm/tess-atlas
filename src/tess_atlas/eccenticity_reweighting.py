# -*- coding: utf-8 -*-
import logging
from astroquery.mast import Catalogs
import numpy as np
from .data import TICEntry
from pymc3.sampling import MultiTrace
import corner
import pandas as pd


def calculate_eccentricity_weights(tic_entry: TICEntry, trace: MultiTrace):
    # TODO: update calculation with https://github.com/exoplanet-dev/tess.world/blob/main/src/tess_world/templates/post.ipynb
    star = Catalogs.query_object(
        f"TIC {tic_entry.tic_number}", catalog="TIC", radius=0.001
    )
    tic_rho_star = float(star["rho"]), float(star["e_rho"])
    logging.info("rho_star = {0} ± {1}".format(*tic_rho_star))

    # Extract the implied density from the fit
    rho_circ = np.repeat(trace["rho_circ"], 100)

    # Sample eccentricity and omega from their priors (the math might
    # be a little more subtle for more informative priors, but I leave
    # that as an exercise for the reader...)
    ecc = np.random.uniform(0, 1, len(rho_circ))
    omega = np.random.uniform(-np.pi, np.pi, len(rho_circ))

    # Compute the "g" parameter from Dawson & Johnson and what true
    # density that implies
    g = (1 + ecc * np.sin(omega)) / np.sqrt(1 - ecc ** 2)
    rho = rho_circ / g ** 3

    # Re-weight these samples to get weighted posterior samples
    log_weights = -0.5 * ((rho - tic_rho_star[0]) / tic_rho_star[1]) ** 2
    weights = np.exp(log_weights - np.max(log_weights))

    # Estimate the expected posterior quantiles
    q = corner.quantile(ecc, [0.16, 0.5, 0.84], weights=weights)
    logging.info(
        "eccentricity = {0:.2f} +{1[1]:.2f} -{1[0]:.2f}".format(
            q[1], np.diff(q)
        )
    )

    return pd.DataFrame(dict(ecc=ecc, omega=omega, weights=weights))