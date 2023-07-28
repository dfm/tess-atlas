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


# + [markdown] tags=["def", "full-width"]
# # TOI Fits
#
# We analysed {{{N_TESS_ATLAS}}} TOIs (TOIs with 2-min cadence lightcurve data availible) out of the {{{N_EXOFOP}}} TOIs.
#
# - {{{N_PASS}}} analyses completed
# - {{{N_FAIL}}} analyses had errors
# - {{{N_NOT_STARTED}}} analyses did not start
#
# Each TOI’s analysis has its own page. Access them from the table below.
#
# + tags=[ "remove-input"]
from itables import init_notebook_mode, show

from tess_atlas.data.analysis_summary import AnalysisSummary

init_notebook_mode(all_interactive=True)
summary_df = AnalysisSummary.from_csv(
    "{{{SUMMARY_PATH}}}"
).generate_summary_table()

# + tags=[ "remove-input"]
show(
    summary_df,
    caption="TESS Atlas Catalog Summary",
    scrollX=True,
    lengthMenu=[5, 10, 20, 50],
    classes="compact",
    maxBytes=0,
)


# + [markdown] tags=["def"]
# The 'classification' column displays the TESS Follow-up Observing Program Working Group (TFOPWG) Dispostion
# of the TOI [[1]]. A full table of the TOIs can be found on the ExoFOP TOI page [[2]].
#
# |    | Description         |
# |----|---------------------|
# | KP | Known Planet        |
# | CP | Confirmed Planet    |
# | PC | Planet Candidate    |
# | APC| Ambiguous Planet Candidate    |
# | FP | False Positive      |
# | FA | False Alarm         |
# | EB | Eclipsing Binary    |
# | IS | Instrument Noise    |
# | V  | Stellar Variability |
# | U  | Undecided           |
# | O  | O                   |
#
# [1]: https://exoplanetarchive.ipac.caltech.edu/docs/API_toi_columns.html
# [2]: https://exofop.ipac.caltech.edu/tess/view_toi.php#
#
