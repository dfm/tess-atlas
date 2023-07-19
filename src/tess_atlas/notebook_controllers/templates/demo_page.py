# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.10.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + [markdown] tags=["def"]
# # Example Loader
#
# Here we demonstrate how to
# 1. Download a TESS-Atlas TOI fit
# 2. Load the stored TOIs data (lightcurve, posterior, stellar parameters)
# 3. Load and make plots of _all_ TOI fits
#
# ## Setup

# + pycharm={"name": "#%%\n"} tags=["def", "hide-cell"]
# %load_ext autoreload
# %load_ext memory_profiler
# %load_ext autotime
# %autoreload 2
# %matplotlib inline

import glob
import logging
import os
from typing import List

import pandas as pd
import tqdm

from tess_atlas.logger import get_notebook_logger
from tess_atlas.plotting import (
    plot_exofop_vs_atlas_comparison,
    plot_toi_list_radius_vs_period,
)
from tess_atlas.utils import notebook_initalisations

OUTDIR = "example_loader_files"


notebook_initalisations()
logger = get_notebook_logger(OUTDIR)


# -

# ## Download a TESS-Atlas fit
#
# To download the notebook and results for a TOI (for example TOI 103) you could use the following:

# ! download_toi 103 --outdir .

# This downloads the notebook along with the results in the specified directory. You can now open that notebook up to load the results or rerun the analysis.
#
# Alternatively, you can also load the results of a TOI (eg TOI 174) in a notebook that you have open with the following:

from tess_atlas.data import TICEntry

toi_174 = TICEntry.load(174, load_from_catalog=True)
toi_174

# ## Download all TESS-Atlas fits
#
# You may want to download _all_ TOI notebooks and fits. To do this, you can use the following.
#
# ```{warning}
# This can take a longgg time!
# ```
#

# ! download_all_tois --outdir tois

# Once you have all the fits you can load them up

# + pycharm={"name": "#%%\n"} tags=["exe"]
SEARCH_PATH = "./*/toi_*_files/*.netcdf"


def get_analysed_toi_sample_filenames() -> pd.DataFrame:
    """Get dataframe of analysed tic number and the path to its samples"""
    files = glob.glob(SEARCH_PATH)
    toi_nums = [
        int("".join(filter(str.isdigit, os.path.basename(f)))) for f in files
    ]
    analysed_tois = pd.DataFrame(dict(toi=toi_nums, path=files))
    logger.info(f"Number of analysed TOIs: {len(analysed_tois)}")
    return analysed_tois


def get_tic_list_with_loaded_samples(
    analysed_tois: List[int],
) -> List[TICEntry]:
    tic_list = [TICEntry.load(toi) for toi in analysed_tois]
    for tic in tqdm.tqdm(tic_list, desc="Loading TIC Samples Sets"):
        tic.load_inference_trace()
    return tic_list


analysed_tois = get_analysed_toi_sample_filenames()
analysed_tois

# + pycharm={"name": "#%%\n"} tags=["exe"]
tic_list = get_tic_list_with_loaded_samples(analysed_tois["toi"])

# + [markdown] tags=["def"]
# To access the loaded samples, access the `.inference_trace` property:

# + pycharm={"name": "#%%\n"} tags=["exe"]
tic_list[0].inference_trace


# + [markdown] tags=["def"]
# Below is a summary of some of the sampled parameter values

# + pycharm={"name": "#%%\n"} tags=["exe"]
summary_df = pd.concat([tic.get_trace_summary() for tic in tic_list])
summary_df
# -

# ## Downloading summary statistics
#
# Maybe you dont want all the fits, but just summary statistics for them. These can be accessed with the following:

# +
from tess_atlas.catalog_stats import get_summary_stats

summary_stats = get_summary_stats()
summary_stats
# -

# ## Plots from all the fits
# Finally, you may want to make some plots with all the results

# +
# make some plots here
## SEE PAPER FIGURES DIR


# -

# # CODE TO GENERATE SUMMARY STATS (shouldnt be here)!!!
# here temporarily so I dont lose the code.
# Ideally should be placed somewhere else
#
# Also, this should be done automatically after TOIs run
#
#

# +
"""
VERY MESSY CODE ATM -- shouldnt be in this notebook but in its _own_ module

Finds all idata objects from pymc3
Reads them and extracts summary info
Saves summary info in a JSON and CSV for future plotting
"""


import json
import re
import warnings
from glob import glob

import arviz as az
import pandas as pd
from arviz import InferenceData
from tqdm.auto import tqdm

R_earth = 6378100
R_sun = 695700000

warnings.filterwarnings("ignore")


EXOFOP = "https://exofop.ipac.caltech.edu/tess/"
TIC_DATASOURCE = EXOFOP + "download_toi.php?sort=toi&output=csv"


def get_exopfop_db():
    db = pd.read_csv(TIC_DATASOURCE)
    print(f"TIC database has {len(db)} entries")
    db[["TOI int", "planet count"]] = (
        db["TOI"].astype(str).str.split(".", 1, expand=True)
    )
    db = db.astype({"TOI int": "int", "planet count": "int"})
    db["Multiplanet System"] = db["TOI int"].duplicated(keep=False)
    db["Single Transit"] = db["Period (days)"] <= 0
    return db


def get_idata_summary(fname):
    inference_data = az.from_netcdf(fname)
    df = az.summary(inference_data, filter_vars="like")

    #     df = (
    #             df.transpose()
    #             .filter(regex=r"(.*p\[.*)|(.*r\[.*)")
    #             .transpose()
    #         )
    df = df[["mean", "sd", "r_hat"]]
    return df.T.to_dict()


def get_pd_summary(fname):
    df = pd.read_csv(fname)
    df = df.describe()
    df = df.filter(regex=r"(.*p\[.*)|(.*r\[.*)").transpose()
    df["sd"] = df["std"]
    df = df[["mean", "sd"]]
    return df.T.to_dict()


def toi_num(f):
    toi_str = re.search(r"toi_(.*\d)", f).group()
    return int(toi_str.split("_")[1])


def get_populoation_summary(summary_info):
    toi_id, p_means, r_means, p_sds, r_sds, rhat_oks = [], [], [], [], [], []
    for toi_num, toi_dat in summary_info.items():
        # get rhat info for TOI
        rhat_ok = True
        for param_dat in toi_dat.values():
            if param_dat["r_hat"] > 1.1:
                rhat_ok = False

        # store each TOI planet's info
        p_id = 0
        while f"r[{p_id}]" in toi_dat:
            toi_id.append(float(f"{toi_num}.{p_id+1:002d}"))
            p_means.append(toi_dat[f"p[{p_id}]"]["mean"])
            p_sds.append(toi_dat[f"p[{p_id}]"]["sd"])
            r_means.append(toi_dat[f"r[{p_id}]"]["mean"])
            r_sds.append(toi_dat[f"r[{p_id}]"]["sd"])
            rhat_oks.append(rhat_ok)
            p_id += 1

    return pd.DataFrame(
        dict(
            TOI=toi_id,
            p_mean=p_means,
            p_sd=p_sds,
            r_mean=r_means,
            r_sd=r_sds,
            rhat_ok=rhat_oks,
        )
    )


def process_all_toi_results():
    root = "july12_cat/0.2.1.dev64+gc7fa3a0/toi_*_files"
    samp_fns = glob(f"{root}/samples.csv")
    idata_fns = glob(f"{root}/inference_data.netcdf")

    print("All results:")
    print(idata_fns)

    # open and summarise _all_ toi results
    summary_info = {}
    for idata_fn in tqdm(idata_fns, desc="Summarising TOI results"):
        summary_info[toi_num(data_fn)] = get_idata_summary(idata_fn)

    with open("summary_stats.json", "w") as outfile:
        json.dump(summary_info, outfile)

    print(f"One TOI summary: {get_pd_summary(samp_fns[0]).T.to_dict()}")
    print("Converting summary to dataframe")
    df = get_populoation_summary(summary_info)

    # download EXOFOP stats
    exofop = get_exopfop_db()
    exofop.to_csv("exofop.csv", index=False)
    exofop = pd.read_csv("exofop.csv")

    # combine exofop results with our results
    combined = pd.merge(right=df, left=exofop, on="TOI", how="outer")
    combined["exo_r"] = (R_earth * combined["Planet Radius (R_Earth)"]) / (
        R_sun * combined["Stellar Radius (R_Sun)"]
    )
    combined.to_csv("combined_res.csv", index=False)
    print("Result summary stored in combined_res.csv")
    return combined


process_all_toi_results()
