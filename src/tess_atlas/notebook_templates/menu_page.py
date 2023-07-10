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
# We analysed {{{N_TOTAL}}} TOIs:
#
# - {{{N_PASS}}} TOIs successfully analysed
# - {{{N_FAIL}}} TOIs had errors
#
# Each TOI’s analysis has its own page. Access them from the table below.
#
# + tags=["exe", "remove-input"]
from itables import init_notebook_mode, show

from tess_atlas.data.analysis_summary import AnalysisSummary

init_notebook_mode(all_interactive=True)
df = AnalysisSummary.load_from_csv("{{{SUMMARY_PATH}}}").dataframe
show(df, caption="TESS Atlas Catalog Summary")
