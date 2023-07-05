# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + [markdown] tags=["def"]
# # TOI Fits
#
# We analysed 2833 TOIs:
#
# - 2746 TOIs successfully analysed
# - 87 TOIs had errors
#
# Each TOI’s analysis has its own page. Here we have a summary of all the TOI fits and links to their pages.
#
# +
from itables import init_notebook_mode, show

from tess_atlas.data.catalog_summary import CatalogSummary

init_notebook_mode(all_interactive=True)
df = CatalogSummary().df
show(df, caption="TESS Atlas Catalog Summary")
