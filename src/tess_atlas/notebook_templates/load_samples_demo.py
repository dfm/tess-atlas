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
# # %load_ext jupyternotify
# %autoreload 2
# %matplotlib inline

import glob
import logging
import os
from typing import List

import pandas as pd
import tqdm


from tess_atlas.utils import get_notebook_logger, notebook_initalisations
from tess_atlas.plotting import (
    plot_toi_list_radius_vs_period,
    plot_exofop_vs_atlas_comparison,
)

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
