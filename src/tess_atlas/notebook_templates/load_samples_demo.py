# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---


# + [markdown] tags=["def"]
# # Notebook to demonstrate procedure to load samples
#
# For a TESS-atlas version, this notebook
# 1. Displays the samples files that are stored
# 2. Loads the stored samples
# 3. Displays the planet params for each toi

# + pycharm={"name": "#%%\n"} tags=["def"]
# %matplotlib inline

import glob
import logging
import os
from typing import List

import pandas as pd
import tqdm

from tess_atlas.data import TICEntry
from tess_atlas.utils import NOTEBOOK_LOGGER_NAME, notebook_initalisations

notebook_initalisations()
logger = logging.getLogger(NOTEBOOK_LOGGER_NAME)

ATLAS_VERSION = "0.2.0"
SEARCH_PATH = "./toi_*_files/*.netcdf"


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


# + pycharm={"name": "#%%\n"} tags=["exe"]
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
