# -*- coding: utf-8 -*-
import logging

import corner
import numpy as np
import pandas as pd
from astroquery.mast import Catalogs
from pymc3.sampling import MultiTrace

from .data import TICEntry


def calculate_eccentricity_weights(tic_entry: TICEntry, trace: MultiTrace):
    star = Catalogs.query_object(
        f"TIC {tic_entry.tic_number}", catalog="TIC", radius=0.001
    )
    tic_rho_star = float(star["rho"]), float(star["e_rho"])

    if not np.all(np.isfinite(tic_rho_star)):
        raise ValueError("The TIC has no measured density")

    logging.info("rho_star = {0} ± {1}".format(*tic_rho_star))

    # Extract the implied density from the fit
    rho_circ = np.repeat(trace["rho_circ"], 500, axis=0)
    period = np.repeat(trace["p"], 500, axis=0)

    # Sample eccentricity and omega uniformly
    ecc = np.random.uniform(0, 1, len(rho_circ))
    omega = np.random.uniform(-np.pi, np.pi, len(rho_circ))

    # Compute the "g" parameter from Dawson & Johnson and what true
    # density that implies
    g = (1 + ecc * np.sin(omega)) / np.sqrt(1 - ecc ** 2)
    rho = rho_circ / g[:, None] ** 3

    # Re-weight these samples to get weighted posterior samples
    log_weights = -0.5 * ((rho - tic_rho_star[0]) / tic_rho_star[1]) ** 2

    list_of_samples_dataframes = []

    for n in range(tic_entry.planet_count):
        # TODO: if single_transit, add samples for np.log10(period[:, n]
        weights = np.exp(log_weights[:, n] - np.max(log_weights[:, n]))
        list_of_samples_dataframes.append(
            pd.DataFrame(
                {
                    f"e[{n}]": ecc,
                    f"omega[{n}]": omega,
                    f"weights[{n}]": weights,
                }
            )
        )

        # Log the expected posterior quantiles
        q = corner.quantile(ecc, [0.16, 0.5, 0.84], weights=weights)
        logging.info(
            f"e[{n}] = {q[1]:.2f} + {np.diff(q)[1]:.2f} - {np.diff(q)[0]:.2f}".format(
                q[1], np.diff(q)
            )
        )

    return pd.concat(list_of_samples_dataframes, axis=1)
