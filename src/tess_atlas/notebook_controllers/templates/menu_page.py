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
# + tags=[ "remove-input", "full-width"]
from itables import init_notebook_mode, show, JavascriptFunction
import itables.options as opt


opt.drawCallback = JavascriptFunction(
    "function(settings) " '{MathJax.Hub.Queue(["Typeset",MathJax.Hub]);}'
)
from tess_atlas.data.analysis_summary import AnalysisSummary

init_notebook_mode(all_interactive=True)
summary_df = AnalysisSummary.load_from_csv("{{{SUMMARY_PATH}}}").generate_summary_table()
show(
    summary_df,
    caption="TESS Atlas Catalog Summary",
    style="width:3500px",
    scrollX=True,
    autoWidth=True,
    lengthMenu=[5, 10, 20, 50],
    classes='cell-border nowrap display compact',
    maxBytes=0,
)
