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
# # TESS Atlas fit for TOI {{{TOINUMBER}}}
#
# **Version: {{{VERSIONNUMBER}}}**
#
# **Note: This notebook was automatically generated as part of the TESS Atlas project. More information can be found on GitHub:** [github.com/dfm/tess-atlas](https://github.com/dfm/tess-atlas)
#
# In this notebook, we do a quicklook fit for the parameters of the TESS Objects of Interest (TOI) in the system number {{{TOINUMBER}}}.
# To do this fit, we use the [exoplanet](https://exoplanet.dfm.io) library and you can find more information about that project at [exoplanet.dfm.io](https://exoplanet.dfm.io).
#
# From here, you can scroll down and take a look at the fit results, or you can:
#
# - [open the notebook in Google Colab to run the fit yourself](https://colab.research.google.com/github/dfm/tess-atlas/blob/gh-pages/notebooks/{{{VERSIONNUMBER}}}/toi-{{{TOINUMBER}}}.ipynb),
# - [view the notebook on GitHub](https://github.com/dfm/tess-atlas/blob/gh-pages/notebooks/{{{VERSIONNUMBER}}}/toi-{{{TOINUMBER}}}.ipynb), or
# - [download the notebook](https://github.com/dfm/tess-atlas/raw/gh-pages/notebooks/{{{VERSIONNUMBER}}}/toi-{{{TOINUMBER}}}.ipynb).
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
# ## Table of Contents
#
# 1. [Getting started](#Getting-started)
# 2. [Data & de-trending](#Data-%26amp%3B-de-trending)
# 3. [Removing stellar variability](#Removing-stellar-variability)
# 4. [Transit model in PyMC3 & exoplanet](#Transit-model-in-PyMC3-%26amp%3B-exoplanet)
# 5. [Sampling](#Sampling)
# 6. [Posterior constraints](#Posterior-constraints)
# 7. [Attribution](#Attribution)
#
# ## Getting started
#
# To get going, we'll need to make out plots show up inline and install a few packages:

# + pycharm={"name": "#%%\n"} tags=["exe"]
# %matplotlib inline

# + [markdown] tags=["def"]
# Then we'll set up the plotting styles and do all of the imports:

# + pycharm={"name": "#%%\n"} tags=["def"]

import logging
import multiprocessing as mp
import os
import warnings
import exoplanet as xo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pymc3 as pm
import pymc3_ext as pmx

from celerite2.theano import GaussianProcess, terms
from pymc3.sampling import MultiTrace

from tess_atlas.eccenticity_reweighting import calculate_eccentricity_weights
from tess_atlas.data import TICEntry
from tess_atlas.plotting import (
    plot_eccentricity_posteriors,
    plot_lightcurve_and_masks,
    plot_lightcurve_with_inital_model,
    plot_masked_lightcurve_flux_vs_time_since_transit,
    plot_posteriors,
)

logging.getLogger().setLevel(logging.INFO)

get_ipython().magic('config InlineBackend.figure_format = "retina"')

# TEMPORARY WORKAROUND
try:
    mp.set_start_method("fork")
except RuntimeError:  # "Multiprocessing context already set"
    pass

# Don't use the schmantzy progress bar
os.environ["EXOPLANET_NO_AUTO_PBAR"] = "true"

# Warning
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Logging setup
logger = logging.getLogger("theano.gof.compilelock")
logger.setLevel(logging.ERROR)
logger = logging.getLogger("exoplanet")
logger.setLevel(logging.DEBUG)

logging.getLogger().setLevel(logging.DEBUG)

# matplotlib settings
plt.style.use("default")
plt.rcParams["savefig.dpi"] = 100
plt.rcParams["figure.dpi"] = 100
plt.rcParams["font.size"] = 16
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Liberation Sans"]
plt.rcParams["font.cursive"] = ["Liberation Sans"]
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["image.cmap"] = "inferno"


# Constants
TOI_DATASOURCE = (
    "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
)

MIN_NUM_DAYS = 0.25

# + pycharm={"name": "#%%\n"} tags=["exe"]
TOI_NUMBER = {{{TOINUMBER}}}
__version__ = {{{VERSIONNUMBER}}}
FILENAME = {{{FILENAME}}}
# -

# ## Fitting stellar parameters
#
# Next, we define some code to grab the TOI list from [ExoFOP](https://exofop.ipac.caltech.edu/tess/) to get the information about the system.
#
# We wrap the information in three objects, a `TIC Entry`, a `Planet Candidate` and finally a `Lightcurve Data` object.
#
# - The `TIC Entry` object holds one or more `Planet Candidate`s (each candidate associated with one TOI id number) and a `Lightcurve Data` for associated with the candidates. Note that the `Lightcurve Data` object is initially the same fopr each candidate but may be masked according to the candidate transit's period.
#
# - The `Planet Candidate` holds informaiton on the TOI data collected by [SPOC](https://heasarc.gsfc.nasa.gov/docs/tess/pipeline.html) (eg transit period, etc)
#
# - The `Lightcurve Data` holds the lightcurve time and flux data for the planet candidates.

# + pycharm={"name": "#%%\n"} tags=["exe"]
tic_entry = TICEntry.generate_tic_from_toi_number(toi=TOI_NUMBER)
tic_entry.display()
# -

# Now, lets download and plot the TESS light curve data for the `Planet Candidate`s using [lightkurve](https://docs.lightkurve.org/):

# + pycharm={"name": "#%%\n"} tags=["exe"]
tic_entry.load_lightcurve()

# + tags=["exe"]
plot_lightcurve_and_masks(tic_entry)
# -

# For efficiency purposes, let's extract just the data within 0.25 days of the transits:

# + pycharm={"name": "#%%\n"} tags=["exe"]
# tic_entry.mask_lightcurve()

# + tags=["exe"]
plot_masked_lightcurve_flux_vs_time_since_transit(tic_entry)


# -

# That looks a little janky, but it's good enough for now.
#
# ## The probabilistic model
#
# We use the probabilistic model as described in [Foreman-Mackey et al 2017] to determine the best parameters to fit the transits present in the lightcurve data.
#
# More explicitly, the stellar light curve $l(t; \vec{\theta})$ is modelled with a Gaussian Process (GP). A GP consists of a mean function $\mu(t;\vec{\theta})$ and a kernel function $k_\alpha(t,t';\vec{\theta})$, where $\vec{\theta}$ is the vector of parameters descibing the lightcurve and $t$ is the time during which the lightcurve is under observation
#
# The parameters describing the lightcurve are
# $\vec{\theta}$ = {
# &emsp;$p_i$ (orbital periods for each planet),
# &emsp;$d_i$ (transit durations for each planet),
# &emsp;$t0_i$ (transit phase/epoch for each planet),
# &emsp;$b_i$ (impact parameter for each planet),
# &emsp;$r_i$ (planet radius in stellar radius for each planet),
# &emsp;$f0$ (baseline relative flux of the light curve from star),
# &emsp;$u1$ $u2$ (two parameters describing the limb-darkening profile of star)
# }
#
# With this we can write
# $$l(t;\vec{\theta}) \sim \mathcal{GP} (\mu(t;\vec{\theta}), k_\alpha(t,t';\vec{\theta}))\ .$$
#
# Here the mean and kernel functions are:
# * $\mu(t;\vec{\theta})$: a limb-darkened transit light curve ([Kipping 2013])
# * $k_\alpha(t,t';\vec{\theta}))$: a stochastically-driven, damped harmonic oscillator ([SHOTterm])
#
#
# Now that we have defined our transit model, we can implement it in python:
#
# [Foreman-Mackey et al 2017]: https://arxiv.org/pdf/1703.09710.pdf
# [Kipping 2013]: https://arxiv.org/abs/1308.0009
# [SHOTterm]: https://celerite2.readthedocs.io/en/latest/api/python/?highlight=SHOTerm#celerite2.terms.SHOTerm

# + pycharm={"name": "#%%\n"} tags=["def"]
def build_planet_transit_model(tic_entry):
    n = tic_entry.planet_count
    t0s = np.array([planet.t0 for planet in tic_entry.candidates])
    depths = np.array([planet.depth for planet in tic_entry.candidates])
    periods = np.array([planet.period for planet in tic_entry.candidates])

    t = tic_entry.lightcurve.time
    y = tic_entry.lightcurve.flux
    yerr = tic_entry.lightcurve.flux_err

    with pm.Model() as my_planet_transit_model:
        ## define planet 𝜃⃗
        t0 = pm.Normal("t0", mu=t0s, sd=1.0, shape=n)
        p = pm.Lognormal("p", mu=np.log(periods), sd=1.0, shape=n)
        d = pm.Lognormal("d", mu=np.log(0.1), sigma=10.0, shape=n)
        r = pm.Lognormal("r", mu=0.5 * np.log(depths * 1e-3), sd=1.0, shape=n)
        b = xo.distributions.ImpactParameter("b", ror=r, shape=n)
        planet_parms = [r, d, b]

        ## define stellar 𝜃⃗
        f0 = pm.Normal("f0", mu=0.0, sd=10.0)
        u = xo.distributions.QuadLimbDark("u")
        stellar_params = [f0, u]

        ## define 𝑘(𝑡,𝑡′;𝜃⃗ )
        jitter = pm.InverseGamma(
            "jitter", **pmx.estimate_inverse_gamma_parameters(1.0, 5.0)
        )
        sigma = pm.InverseGamma(
            "sigma", **pmx.estimate_inverse_gamma_parameters(1.0, 5.0)
        )
        rho = pm.InverseGamma(
            "rho", **pmx.estimate_inverse_gamma_parameters(0.5, 10.0)
        )
        kernel = terms.SHOTerm(sigma=sigma, rho=rho, Q=0.3)
        noise_params = [jitter, sigma, rho]

        ## define 𝜇(𝑡;𝜃) (ie light)
        orbit = xo.orbits.KeplerianOrbit(
            period=p, t0=t0, b=b, duration=d, ror=r
        )
        lightcurves = xo.LimbDarkLightCurve(u).get_light_curve(
            orbit=orbit, r=r, t=t
        )
        lightcurve = 1e3 * pm.math.sum(lightcurves, axis=-1) + f0
        lightcurves = pm.Deterministic("lightcurves", lightcurves)
        rho_circ = pm.Deterministic("rho_circ", orbit.rho_star)

        # Finally the GP observation model
        residual = y - lightcurve
        gp = GaussianProcess(
            kernel, t=t, diag=yerr ** 2 + jitter ** 2, mean=lightcurve
        )
        gp.marginal("obs", observed=y)

        # cache params
        my_params = dict(
            planet_params=planet_parms,
            noise_params=noise_params,
            stellar_params=stellar_params,
        )
    return my_planet_transit_model, my_params, gp


def test_model(model):
    """Test a point in the model and assure no nans"""
    with model:
        test_prob = model.check_test_point()
        test_prob.name = "log P(test-point)"
        assert not test_prob.isnull().values.any(), test_prob
        test_pt = pd.Series(
            {
                k: str(round(np.array(v).flatten()[0], 2))
                for k, v in model.test_point.items()
            },
            name="Test Point",
        )
        return pd.concat([test_pt, test_prob], axis=1)


# + pycharm={"name": "#%%\n"} tags=["exe"]
planet_transit_model, params, gp = build_planet_transit_model(tic_entry)
test_model(planet_transit_model)


# -

# The test point acts as an example of a point in the parameter space.
# We can now optimize the model sampling parameters before initialising the sampler.

# + tags=["def"]
def get_optimized_init_params(
    model, planet_params, noise_params, stellar_params
):
    """Get params with maximimal log prob for sampling starting point"""
    logging.info("Optimizing sampling starting point")
    with model:
        theta = model.test_point
        kwargs = dict(start=theta, verbose=False, progress=False)
        theta = pmx.optimize(**kwargs, vars=[noise_params[0]])
        theta = pmx.optimize(**kwargs, vars=planet_params)
        theta = pmx.optimize(**kwargs, vars=noise_params)
        theta = pmx.optimize(**kwargs, vars=stellar_params)
    logging.info("Optimization complete!")
    return theta


# + tags=["exe"]
init_params = get_optimized_init_params(planet_transit_model, **params)
# -

# Now we can plot our initial model:

# + pycharm={"name": "#%%\n"} tags=["exe"]
plot_lightcurve_with_inital_model(tic_entry, init_params)

# + tags=["exe"]
plot_masked_lightcurve_flux_vs_time_since_transit(
    tic_entry=tic_entry,
    model_lightcurves=[
        init_params["lightcurves"][:, i] * 1e3
        for i in range(tic_entry.planet_count)
    ],
)
# -

# That looks better!
#
# Now on to sampling:

# + pycharm={"name": "#%%\n"} tags=["def"]
TUNE = 2000
DRAWS = 2000


def start_model_sampling(model) -> MultiTrace:
    np.random.seed(TOI_NUMBER)

    with model:
        samples = pmx.sample(
            tune=TUNE, draws=DRAWS, start=init_params, chains=2, cores=1
        )
        return samples


# + pycharm={"name": "#%%\n"} tags=["exe"]
trace = start_model_sampling(planet_transit_model)
# -

# Then we can take a look at the summary statistics:

# + pycharm={"name": "#%%\n"} tags=["exe"]
tic_entry.inference_trace = trace
tic_entry.inference_trace


# -

# And plot the posterior probability distributuions:


# + pycharm={"name": "#%%\n"} tags=["exe"]
plot_posteriors(tic_entry, trace)


# -

# Finally, we save the posteriors and sampling metadata for future use.

# + pycharm={"name": "#%%\n"} tags=["exe"]
tic_entry.save_inference_trace()


# -

# ## Eccentricity
#
# As discussed above, we fit this model assuming a circular orbit which speeds things up for a few reasons:
# 1) `e=0` allows simpler orbital dynamics which are more computationally efficient (no need to solve Kepler's equation numerically)
# 2) There are degeneracies between eccentricity, arrgument of periasteron, impact parameter, and planet radius. Hence by setting `e=0` and using the duration in calculating the planet's orbit, the sampler can perform better.
#
# But, in this case, the planet *is* actually on an eccentric orbit, so that assumption isn't justified.
# It has been recognized by various researchers over the years (I first learned about this from [Bekki Dawson](https://arxiv.org/abs/1203.5537)) that, to first order, the eccentricity mainly just changes the transit duration.
# The key realization is that this can be thought of as a change in the impled density of the star.
# Therefore, if you fit the transit using stellar density (or duration, in this case) as one of the parameters (*note: you must have a* different *stellar density parameter for each planet if there are more than one*), you can use an independent measurement of the stellar density to infer the eccentricity of the orbit after the fact.
# All the details are described in [Dawson & Johnson (2012)](https://arxiv.org/abs/1203.5537), but here's how you can do this here using the stellar density listed in the TESS input catalog:

# + pycharm={"name": "#%%\n"} tags=["exe"]
ecc_samples = calculate_eccentricity_weights(tic_entry, trace)
ecc_samples.to_csv(os.path.join(tic_entry.outdir, "eccentricity_samples.csv"))
plot_eccentricity_posteriors(tic_entry, ecc_samples)
# -

# As you can see, this eccentricity estimate is consistent (albeit with large uncertainties) with the value that [Pepper et al. (2019)](https://arxiv.org/abs/1911.05150) measure using radial velocities and it is definitely clear that this planet is not on a circular orbit.

# ## Citations
#
# As described in the :ref:`citation` tutorial, we can use :func:`exoplanet.citations.get_citations_for_model` to construct an acknowledgement and BibTeX listing that includes the relevant citations for this model.

# + pycharm={"name": "#%%\n"} tags=["exe"]
with planet_transit_model:
    txt, bib = xo.citations.get_citations_for_model()
print(txt)

# + pycharm={"name": "#%%\n"} tags=["exe"]
print("\n".join(bib.splitlines()[:10]) + "\n...")
