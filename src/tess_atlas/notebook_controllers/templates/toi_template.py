# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   orphan: true
# ---

# + tags=["exe", "hide-cell"]
# ! pip install tess-atlas -q

from tess_atlas.utils import notebook_initalisations

notebook_initalisations()

# + [markdown] tags=["def"]
# # TESS Atlas fit for TOI {{{TOINUMBER}}}
#
# **Version: {{{VERSIONNUMBER}}}**
#
# **Note: This notebook was automatically generated as part of the TESS Atlas project. More information can be found on GitHub:** [github.com/dfm/tess-atlas](https://github.com/dfm/tess-atlas)
#
# In this notebook, we do a quicklook fit for the parameters of the TESS Objects of Interest (TOI) in the system number {{{TOINUMBER}}}.
# To do this fit, we use the [exoplanet](https://exoplanet.dfm.io) library and you can find more information about that project at [exoplanet.dfm.io](https://exoplanet.dfm.io).
#
# From here, you can
# - scroll down and take a look at the fit results
# - open the notebook in Google Colab to run the fit yourself
# - __download the notebook
#
#
#
# ## Caveats
#
# There are many caveats associated with this relatively simple "quicklook" type of analysis that should be kept in mind.
# Here are some of the main things that come to mind:
#
# 1. The orbits that we fit are constrained to be *circular*. One major effect of this approximation is that the fit will significantly overestimate the confidence of the impact parameter constraint, so the results for impact parameter shouldn't be taken too seriously.
#
# 2. Transit timing variations, correlated noise, and (probably) your favorite systematics are ignored. Sorry!
#
# 3. This notebook was generated automatically without human intervention. Use at your own risk!
#
#
# ## Getting started
#
# To get going, we'll add some _magic_, import some packages, and run some setup steps.

# + pycharm={"name": "#%%\n"} tags=["def", "hide-cell", "remove-output"]
# %load_ext autoreload
# %load_ext memory_profiler
# %load_ext autotime
# %autoreload 2
# # %matplotlib inline

import os

import numpy as np
import pymc3_ext as pmx
from arviz import InferenceData

from tess_atlas.analysis.eccenticity_reweighting import (
    calculate_eccentricity_weights,
)
from tess_atlas.analysis.model_tools import (
    compute_variable,
    get_untransformed_varnames,
    sample_prior,
)
from tess_atlas.data.inference_data_tools import (
    get_optimized_init_params,
    summary,
    test_model,
)
from tess_atlas.data.tic_entry import TICEntry
from tess_atlas.logger import get_notebook_logger
from tess_atlas.plotting import (
    plot_diagnostics,
    plot_eccentricity_posteriors,
    plot_inference_trace,
    plot_lightcurve,
    plot_phase,
    plot_posteriors,
    plot_priors,
    plot_raw_lightcurve,
)

TOI_NUMBER = {{{TOINUMBER}}}
logger = get_notebook_logger(outdir=f"toi_{TOI_NUMBER}_files")

# + tags=["exe", "remove-cell"]
import theano

from tess_atlas.utils import tabulate_global_environ_vars

logger.info("Logging some settings for future reference")
logger.info("GLOBAL ENVS:\n" + tabulate_global_environ_vars())
logger.info(f"THEANO Config:\n{theano.config}")

# + [markdown] tags=["def"]
# ## Downloading Data
#
# Next, we grab some inital guesses for the TOI's parameters from [ExoFOP](https://exofop.ipac.caltech.edu/tess/) and __download the TOI's lightcurve with [Lightkurve].
#
# We wrap the information in three objects, a `TIC Entry`, a `Planet Candidate` and finally a `Lightcurve Data` object.
#
# - The `TIC Entry` object holds one or more `Planet Candidate`s (each candidate associated with one TOI id number) and a `Lightcurve Data` for associated with the candidates. Note that the `Lightcurve Data` object is initially the same fopr each candidate but may be masked according to the candidate transit's period.
#
# - The `Planet Candidate` holds information on the TOI data collected by [SPOC] (eg transit period, etc)
#
# - The `Lightcurve Data` holds the lightcurve time and flux data for the planet candidates.
#
# [ExoFOP]: https://exofop.ipac.caltech.edu/tess/
# [Lightkurve]: https://docs.lightkurve.org/index.html
# [SPOC]: https://heasarc.gsfc.nasa.gov/docs/tess/pipeline.html
#
# Downloading the data (this may take a few minutes):
# + pycharm={"name": "#%%\n"} tags=["exe"]
tic_entry = TICEntry.load(toi=TOI_NUMBER)
tic_entry
# -

# If the amount of lightcurve data availible is large we filter the data to keep only data around transits.

# + tags=["exe"]
if tic_entry.lightcurve.len > 1e5:
    tic_entry.lightcurve.filter_non_transit_data(tic_entry.candidates)
else:
    logger.info("Using the full lightcurve for analysis.")
# -

# Plot of the lightcurve:


# + pycharm={"name": "#%%\n"} tags=["exe", "remove-output"]
plot_lightcurve(tic_entry, save=True)

# Some diagnostics
plot_raw_lightcurve(tic_entry, save=True)
plot_raw_lightcurve(tic_entry, zoom_in=True, save=True)
# -
# ![](toi_{{{TOINUMBER}}}_files/flux_vs_time.png)

# + [markdown] tags=["hide-cell"]
# Diagnostic plots of the raw lightcurve (not applying sigma clipping/other cleaning methods to remove outliers). Some things to consider:
# - Do the initial fits from ExoFOP match the transits if visible?
# - If this is marked as a single-transit event, is there only 1 transit visible?
#
# ![](toi_{{{TOINUMBER}}}_files/diagnostic_raw_lc_plot.png)
# ![](toi_{{{TOINUMBER}}}_files/diagnostic_raw_lc_plot_zoom.png)
# -

# ## Fitting transit parameters
# Now that we have the data, we can define a Bayesian model to fit it.
#
# ### The probabilistic model
#
# We use the probabilistic model as described in [Foreman-Mackey et al 2017] to determine the best parameters to fit the transits present in the lightcurve data.
#
# More explicitly, the stellar light curve $l(t; \vec{\theta})$ is modelled with a Gaussian Process (GP).
# A GP consists of a mean function $\mu(t;\vec{\theta})$ and a kernel function $k_\alpha(t,t';\vec{\theta})$, where $\vec{\theta}$ is the vector of parameters descibing the lightcurve and $t$ is the time during which the lightcurve is under observation
#
# The 8 parameters describing the lightcurve are
# $$\vec{\theta} = \{d_i, t0_i, tmax_i, b_i, r_i, f0, u1, u2\},$$
# where
# * $d_i$ transit durations for each planet,
# * $tmin_i$ time of first transit for each planet (reference time),
# * $tmax_i$ time of the last transit for each planet (a second reference time),
# * $b_i$ impact parameter for each planet,
# * $r_i$ planet radius in stellar radius for each planet,
# * $f0$ baseline relative flux of the light curve from star,
# * $u1$ $u2$ two parameters describing the limb-darkening profile of star.
#
# Note: if the observed data only records a single transit,
# we swap $tmax_i$ with $p_i$ (orbital periods for each planet).
#
# With this we can write
# $$l(t;\vec{\theta}) \sim \mathcal{GP} (\mu(t;\vec{\theta}), k_\alpha(t,t';\vec{\theta}))\ .$$
#
# Here the mean and kernel functions are:
# * $\mu(t;\vec{\theta})$: a limb-darkened transit light curve ([Kipping 2013])
# * $k_\alpha(t,t';\vec{\theta}))$: a stochastically-driven, damped harmonic oscillator ([SHOTterm])
#
#
# Now that we have defined our transit model, we can implement it in python (toggle to show).
#
# [Foreman-Mackey et al 2017]: https://arxiv.org/pdf/1703.09710.pdf
# [Kipping 2013]: https://arxiv.org/abs/1308.0009
# [SHOTterm]: https://celerite2.readthedocs.io/en/latest/api/python/?highlight=SHOTerm#celerite2.terms.SHOTerm

# + pycharm={"name": "#%%\n"} tags=["def", "hide-cell"]
{{{TRANSIT_MODEL_CODE}}}

# + pycharm={"name": "#%%\n"} tags=["exe"]
planet_transit_model, params = build_planet_transit_model(tic_entry)
model_varnames = get_untransformed_varnames(planet_transit_model)
test_model(planet_transit_model)


# -

# ### Optimizing the initial point for sampling
# We help out the sampler we try to find an optimized set of initial parameters to begin sampling from.

# + tags=["exe", "remove-output"]
if tic_entry.optimized_params is None:
    init_params = get_optimized_init_params(planet_transit_model, **params)
    tic_entry.save_data(optimized_params=init_params)
else:
    init_params = tic_entry.optimized_params.to_dict()

# + tags=["exe"]
# sanity check that none of the right hand column have nans!
test_model(planet_transit_model, init_params, show_summary=True)
# -

# Below are plots of our initial model and priors.
#
# ### Initial model fit

# + pycharm={"name": "#%%\n"} tags=["exe", "remove-output"]
initial_lc_models = (
    compute_variable(
        model=planet_transit_model,
        samples=[[init_params[n] for n in model_varnames]],
        target=planet_transit_model.lightcurve_models,
    )
    * 1e3
)
plot_lightcurve(
    tic_entry, initial_lc_models, save="lightcurve_with_initial_guess.png"
)
plot_lightcurve(
    tic_entry,
    initial_lc_models,
    zoom_in=True,
    save="lightcurve_with_initial_guess_zoom.png",
)
# -

# <!-- Show LC plot with initial guess -->
# ![](toi_{{{TOINUMBER}}}_files/lightcurve_with_initial_guess.png)

# + [markdown] tags=["hide-cell"]
# <!-- Show zoomed in LC plot with initial guess -->
# ![](toi_{{{TOINUMBER}}}_files/lightcurve_with_initial_guess_zoom.png)

# + tags=["exe", "remove-output"]
params = dict(
    tic_entry=tic_entry, model=planet_transit_model, initial_params=init_params
)
plot_phase(**params, save="phase_initial.png")
plot_phase(
    **params, plot_all_datapoints=True, save="phase_initial_all_datapoints.png"
)
# -

# <!-- SHOW PHASE PLOT -->
# <img src="toi_{{{TOINUMBER}}}_files/phase_initial.png" style="width:450px;"/>
#

# + [markdown] tags=["hide-cell"]
# Diagnostic phase plot
# <img src="toi_{{{TOINUMBER}}}_files/phase_initial_all_datapoints.png" style="width:450px;"/>
# -

# ### Histograms of Priors

# + tags=["exe", "remove-output"]
prior_samples = sample_prior(planet_transit_model)
if prior_samples:
    plot_priors(tic_entry, prior_samples, init_params, save=True)


# -

# ![](toi_{{{TOINUMBER}}}_files/priors.png)

#
# ### Sampling
# With the model and priors defined, we can begin sampling.


# + pycharm={"name": "#%%\n"} tags=["def"]
def run_inference(model) -> InferenceData:
    np.random.seed(TOI_NUMBER)
    with model:
        sampling_kwargs = dict(tune=2000, draws=2000, chains=2, cores=2)
        logger.info(f"Run sampler with kwargs: {sampling_kwargs}")
        inference_data = pmx.sample(
            **sampling_kwargs, start=init_params, return_inferencedata=True
        )
        logger.info("Sampling completed!")
        return inference_data


# + pycharm={"name": "#%%\n"} tags=["exe"]
if tic_entry.inference_data is None:
    inference_data = run_inference(planet_transit_model)
    tic_entry.inference_data = inference_data
    tic_entry.save_data(inference_data=inference_data)
else:
    logger.info("Using cached run")
    inference_data = tic_entry.inference_data
inference_data

# -

# The `inference_data` object contains the posteriors and sampling metadata. Let's save it for future use, and take a look at summary statistics. Note: the _trace plot_ from sampling is hidden below.

# + pycharm={"name": "#%%\n"} tags=["exe"]
summary(inference_data)
# + tags=["exe", "remove-output"]
plot_inference_trace(tic_entry, save=True)

# + [markdown] tags=["hide-cell"]
# ![](toi_{{{TOINUMBER}}}_files/diagnostic_trace_plot.png)
# -

# ## Results
#
#
# ### Posterior plots
# Below are plots of the posterior probability distributions and the best-fitting light-curve model.

# + pycharm={"name": "#%%\n"} tags=["exe", "remove-output"]
plot_posteriors(
    tic_entry, inference_data, initial_params=init_params, save=True
)
# -
# <!-- SHOW POSTERIOR PLOT -->
# ![](toi_{{{TOINUMBER}}}_files/posteriors.png)

# + pycharm={"name": "#%%\n"} tags=["exe", "remove-output"]
# %%memit
plot_phase(
    tic_entry,
    planet_transit_model,
    inference_data,
    initial_params=init_params,
    save=True,
)
# -

# <!-- SHOW PHASE PLOT -->
# <img src="toi_{{{TOINUMBER}}}_files/phase_plot.png" style="width:450px;"/>

# ### Eccentricity post-processing
#
# As discussed above, we fit this model assuming a circular orbit which speeds things up for a few reasons:
# 1) `e=0` allows simpler orbital dynamics which are more computationally efficient (no need to solve Kepler's equation numerically)
#
# 2) There are degeneracies between eccentricity, arrgument of periasteron, impact parameter, and planet radius. Hence by setting `e=0` and using the duration in calculating the planet's orbit, the sampler can perform better.
#
# To first order, the eccentricity mainly just changes the transit duration.
# This can be thought of as a change in the impled density of the star.
# Therefore, if the transit is fit using stellar density (or duration, in this case) as one of the parameters, it is possible to make an independent measurement of the stellar density, and in turn infer the eccentricity of the orbit as a post-processing step.
# The details of this eccentricity calculation method are described in [Dawson & Johnson (2012)].
#
# Here, if the TIC has associated stellar data, we use the method described above to obtain fits for the exoplanet's orbital eccentricity.
#
# [Dawson & Johnson (2012)]: https://arxiv.org/abs/1203.5537
# Note: a different stellar density parameter is required for each planet (if there is more than one planet)

# + pycharm={"name": "#%%\n"} tags=["exe"]
star = tic_entry.stellar_data
star

# + tags=["remove-input"]
if star.density_data_present:
    logger.info(
        "Stellar data present for TIC. Continuing with eccentricity calculations."
    )
else:
    logger.info(
        "Stellar data not present for TIC. Skipping eccentricity calculations."
    )

# + pycharm={"name": "#%%\n"} tags=["exe", "remove-output"]
if star.density_data_present:
    ecc_samples = calculate_eccentricity_weights(tic_entry, inference_data)
    ecc_samples.to_csv(
        os.path.join(tic_entry.outdir, "eccentricity_samples.csv"), index=False
    )
    plot_eccentricity_posteriors(tic_entry, ecc_samples, save=True)
# -
# <!-- SHOW ECC POSTERIORS -->
# ![](toi_{{{TOINUMBER}}}_files/eccentricity_posteriors.png)

# ### Diagnostics
# Finally, we also generate some diagnostic plots.

# + pycharm={"name": "#%%\n"} tags=["exe", "remove-output"]
plot_diagnostics(tic_entry, planet_transit_model, init_params, save=True)
# + [markdown] tags=["hide-cell"]
# <!-- SHOW DIAGNOSTICS -->
#
# ![](toi_{{{TOINUMBER}}}_files/diagnostic_flux_vs_time_zoom.png)
# -

# ## Citations
#
# We hope this has been helpful! The TESS-Atlas was built using exoplanet, PyMC3, lightkurve, starry, celerite2, ExoFOP, and Sphinx.
#
# We would greatly appreciate you citing this work and its dependencies.
#
# ### LaTeX acknowledgement and bibliography

# + pycharm={"name": "#%%\n"} tags=["exe", "output_scroll"]
from tess_atlas import citations

citations.print_acknowledgements()

# + pycharm={"name": "#%%\n"} tags=["exe", "output_scroll"]
citations.print_bibliography()
# -

# ### Packages used
#

# + pycharm={"name": "#%%\n"} tags=["exe", "output_scroll"]
citations.print_packages()
# -

# ## Comments
# Leave a comment below or in this [issue](https://github.com/avivajpeyi/tess-atlas/issues/new?title=TOI{{{TOINUMBER}}}).
# ```{raw} html
# <script src="https://utteranc.es/client.js"
#         repo="avivajpeyi/tess-atlas"
#         issue-term="TOI{{{TOINUMBER}}}"
#         theme="github-light"
#         crossorigin="anonymous"
#         async>
# </script>
# ```
