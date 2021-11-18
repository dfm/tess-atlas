"""MWE to download data needed for TESS atlas jobs"""
import ssl

import certifi
import numpy as np
import pandas as pd

EXOFOP = "https://exofop.ipac.caltech.edu/tess/"
TIC_DATASOURCE = EXOFOP + "download_toi.php?sort=toi&output=csv"
TIC_NUM = 336732616


def get_data_table():
    print("Downloading table from Exofop website")
    db = pd.read_csv(TIC_DATASOURCE)
    print("Completed getting TIC datasource")


def get_stellar_data():
    """Gets stellar information for TIC"""
    from astroquery.mast import (  # loading here so the script doesnt when testing base python (w/o astroquery installed)
        Catalogs,
    )

    print("Starting astroquery")
    star = Catalogs.query_object(
        f"TIC {TIC_NUM}", catalog="TIC", radius=0.001
    )[
        0
    ]  # only selecting the 1st row
    star = dict(rho=float(star["rho"]), e_rho=float(star["e_rho"]))
    print(f"rho_star = {star['rho']} ± {star['e_rho']}")
    print("Completed astroquery")


def get_lk_data():
    import lightkurve as lk  # loading here just so the script doesnt die when using base python (w/o lk installed)

    search = lk.search_lightcurve(target=f"TIC {TIC_NUM}", mission="TESS")
    # Restrict to short cadence no "fast" cadence
    search = search[np.where(search.table["t_exptime"] == 120)]
    print(
        f"Downloading {len(search)} observations of light curve data "
        f"(TIC {TIC_NUM})"
    )
    data = search.download_all()
    print("Completed downloading lightcurve data")


def main():
    main_cert = ssl.get_default_verify_paths().openssl_cafile
    current_cert = certifi.where()
    print(f"Main: {main_cert}\nCurrent:{current_cert}")
    get_data_table()
    try:
        get_lk_data()
        get_stellar_data()
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
