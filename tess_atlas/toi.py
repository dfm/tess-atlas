# -*- coding: utf-8 -*-

__all__ = [
    "get_toi_list",
    "get_info_for_toi",
    "get_info_for_tic",
    "get_gaia_data_for_toi",
    "get_gaia_data_for_tic",
    "fit_gaia_data_for_toi",
    "fit_gaia_data_for_tic",
]

import numpy as np
import pandas as pd
from astroquery.mast import Catalogs

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.utils.data import download_file

from .stellar import fit_gaia_data, get_gaia_data
from .tess_atlas_version import __version__

TOI_URL = "https://exofop.ipac.caltech.edu/tess/download_toi.php"


def get_toi_list(use_cache=True):
    url = TOI_URL + "?sort=toi&output=csv&version={0}".format(__version__)
    if use_cache:
        return pd.read_csv(download_file(url, cache=True))
    return pd.read_csv(url)


def get_info_for_toi(toi_num, use_cache=True):
    tois = get_toi_list(use_cache=use_cache)

    # Select all of the rows in the table that are associated with this target
    toi = tois[tois["TOI"] == int(toi_num) + 0.01].iloc[0]
    tic = toi["TIC ID"]
    tois = tois[tois["TIC ID"] == tic].sort_values("TOI")

    # Extract the planet periods
    periods = np.array(tois["Period (days)"], dtype=float)
    assert np.all(periods > 0), "We haven't implemented single transits yet"

    # Convert the phase to TBJD from BJD
    tois["t0s"] = np.array(tois["Epoch (BJD)"], dtype=float) - 2457000

    # Convert the depth to parts per thousand from parts per million
    tois["depths"] = 1e-3 * np.array(tois["Depth (ppm)"], dtype=float)

    # Convert the duration to days from hours
    tois["durations"] = np.array(tois["Duration (hours)"], dtype=float) / 24.0

    return tois


def get_info_for_tic(tic):
    r = Catalogs.query_object(
        "TIC {0}".format(tic), radius=20 * u.arcsec, catalog="TIC"
    )
    return r[r["ID"] == "{0}".format(tic)]


def get_gaia_data_for_toi(toi_num, use_cache=True, **kwargs):
    info = get_info_for_toi(toi_num, use_cache=use_cache)
    coord = SkyCoord(
        ra=info["RA"][0], dec=info["Dec"][0], unit=(u.hourangle, u.deg)
    )
    return get_gaia_data(
        coord, approx_mag=float(info["TESS Mag"][0]), **kwargs
    )


def get_gaia_data_for_tic(tic, **kwargs):
    info = get_info_for_tic(tic)
    coord = SkyCoord(
        ra=float(info["ra"]) * u.deg, dec=float(info["dec"]) * u.deg
    )
    return get_gaia_data(coord, approx_mag=float(info["GAIAmag"]), **kwargs)


def fit_gaia_data_for_toi(toi_num, clobber=False, **kwargs):
    name = "toi{0}".format(toi_num)
    data = get_gaia_data_for_toi(toi_num, **kwargs)
    return fit_gaia_data(name, data, clobber=clobber)


def fit_gaia_data_for_tic(tic, clobber=False, **kwargs):
    name = "tic{0}".format(tic)
    data = get_gaia_data_for_tic(tic, **kwargs)
    return fit_gaia_data(name, data, clobber=clobber)
