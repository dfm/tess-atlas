import json
import logging
import os
import time
from typing import Dict

import numpy as np
from astroquery.exceptions import NoResultsWarning, ResolverError
from astroquery.mast import Catalogs
from IPython.display import HTML, display
from requests.models import HTTPError

from tess_atlas.utils import NOTEBOOK_LOGGER_NAME

from .data_object import DataObject

logger = logging.getLogger(NOTEBOOK_LOGGER_NAME)


class StellarData(DataObject):
    def __init__(
        self,
        density: float,
        density_error: float,
        radius: float,
        radius_error: float,
        mass: float,
        mass_error: float,
        outdir: str,
    ):
        """
        :param float density: (units: Solar)
        :param float density_error: uncertainty in density (units: Solar)
        :param float radius: Stellar radius derived from photometry ( (units: Solar Radii)
        :param float mass: Stellar mass derived from photometry (units: Solar Mass)
        """
        self.density = density
        self.density_error = density_error
        self.mass = mass
        self.mass_error = mass_error
        self.radius = radius
        self.radius_error = radius_error
        self.outdir = outdir

    @classmethod
    def from_database(cls, tic: int, outdir: str):
        """
        MAST has info on the TIC's associated stellar info.
        Properties of the catalog are here: https://arxiv.org/pdf/1905.10694.pdf
        """
        logger.info(f"Downloading StellarData from MAST")
        try:
            star = get_tic_stellar_data_from_mast(tic)
        except (ResolverError, NoResultsWarning):
            logger.warning(
                f"No stellar data associated with TIC {tic}. Eccentricity post-processing step may fail."
            )
            star = dict()
        return cls(
            density=star.get("rho", np.nan),
            density_error=star.get("rho", np.nan),
            mass=star.get("mass", np.nan),
            mass_error=star.get("e_mass", np.nan),
            radius=star.get("radius", np.nan),
            radius_error=star.get("e_radius", np.nan),
            outdir=outdir,
        )

    @classmethod
    def from_cache(cls, tic: int, outdir: str):
        fpath = StellarData.get_filepath(outdir)
        with open(fpath, "r") as f:
            data = json.load(f)
        logger.info(f"StellarData loaded from {fpath}")
        return cls(**data, outdir=outdir)

    def save_data(self, outdir):
        fpath = self.get_filepath(outdir)
        with open(fpath, "w") as f:
            json.dump(self.to_dict(), fp=f, indent=2)

    def density_data_present(self):
        return no_nans_present([self.density, self.density_error])

    def __str__(self):
        stellar_data = ["Stellar Info: "]
        if no_nans_present([self.mass, self.mass_error]):
            stellar_data.append(
                f"mass = {self.mass:0.2f} ± {self.mass_error:0.2f} (Msun)"
            )
        if no_nans_present([self.radius, self.radius_error]):
            stellar_data.append(
                f"radius = {self.radius:0.2f} ± {self.radius_error:0.2f} (Rsun)"
            )
        if self.density_data_present():
            stellar_data.append(
                f"density = {self.density:0.2f} ± {self.density_error:0.2f} (solar)"
            )
        if len(stellar_data) == 1:
            stellar_data.append("No data present")
        return "\n".join(stellar_data)

    def display(self):
        str_list = self.__str__().split("\n")
        new_str = [f"<b>{str_list[0]}</b>", "<ul>"]
        for i in range(1, len(str_list)):
            new_str.append(f"<li>{str_list[i]}</li>")
        new_str.append("</ul>")
        return display(HTML("\n".join(new_str)))

    def to_dict(self):
        return dict(
            density=self.density,
            density_error=self.density_error,
            mass=self.mass,
            mass_error=self.mass_error,
            radius=self.radius,
            radius_error=self.radius_error,
        )

    @staticmethod
    def get_filepath(outdir: str, fname="stellar_data.json") -> str:
        return os.path.join(outdir, fname)


def no_nans_present(data):
    return np.all(np.isfinite(data))


def get_tic_stellar_data_from_mast(
    tic: int, num_retry=3, wait_time=0.2
) -> Dict[str, float]:
    logger.debug(
        f"astroquery.mast.Catalogs.query_object('TIC {tic}', catalog='TIC', radius=0.001)"
    )
    for attempt_number in range(num_retry):
        try:
            astropy_star_datatable = Catalogs.query_object(
                f"TIC {tic}", catalog="TIC", radius=0.001
            )
            logger.debug(f"Downloading StellarData from MAST successful")
            df = astropy_star_datatable.to_pandas()
            star = df.to_dict("records")[0]  # only selecting the 1st as a dict
            return star
        except HTTPError as e:
            if attempt_number < num_retry - 1:
                logger.warning(
                    f"astroquery.mast call (attempt {attempt_number}) failed: {e}. "
                    f"Retrying after {wait_time}s."
                )
                time.sleep(wait_time)
                continue
    logger.error(
        f"astroquery.mast call failed. Eccentricity post-processing will fail."
    )
    return {}
