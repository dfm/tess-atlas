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
# 2. [Downloading Data](#Downloading-Data)
# 3. [Fitting stellar parameters](#Fitting-stellar-parameters)
# 4. [Results](#Results)
# 5. [Citations](#Citations)
# 6. [Posterior constraints](#Posterior-constraints)
# 7. [Attribution](#Attribution)
#
# ## Getting started
#
# To get going, we'll need to make out plots show up inline:

# + pycharm={"name": "#%%\n"} tags=["exe"]
# %matplotlib inline

# + [markdown] tags=["def"]
# Then we'll set up the plotting styles and do all of the imports:

# + pycharm={"name": "#%%\n"} tags=["def"]
from tess_atlas.utils import notebook_initalisations

notebook_initalisations()

# + pycharm={"name": "#%%\n"} tags=["exe"]
import os
import exoplanet as xo
import numpy as np
import pandas as pd
import pymc3 as pm
import pymc3_ext as pmx
import aesara_theano_fallback.tensor as tt

from celerite2.theano import GaussianProcess, terms
from pymc3.sampling import MultiTrace

from tess_atlas.data import TICEntry
from tess_atlas.analysis.eccenticity_reweighting import (
    calculate_eccentricity_weights,
)
from tess_atlas.utils import get_notebook_logger


# + pycharm={"name": "#%%\n"} tags=["exe"]
os.environ["INTERACTIVE_PLOTS"] = "FALSE"  # "TRUE" for interactive plots
from tess_atlas.plotting import (
    plot_eccentricity_posteriors,
    plot_folded_lightcurve,
    plot_phase,
    plot_lightcurve,
    plot_posteriors,
)

# + pycharm={"name": "#%%\n"} tags=["exe"]
TOI_NUMBER = {{{TOINUMBER}}}
logger = get_notebook_logger(outdir=f"toi_{TOI_NUMBER}_files")

# + [markdown] tags=["def"]
# ## Downloading Data
#
# Next, we grab some inital guesses for the TOI's parameters from [ExoFOP](https://exofop.ipac.caltech.edu/tess/) and download the TOI's lightcurve with [Lightkurve].
#
# We wrap the information in three objects, a `TIC Entry`, a `Planet Candidate` and finally a `Lightcurve Data` object.
#
# - The `TIC Entry` object holds one or more `Planet Candidate`s (each candidate associated with one TOI id number) and a `Lightcurve Data` for associated with the candidates. Note that the `Lightcurve Data` object is initially the same fopr each candidate but may be masked according to the candidate transit's period.
#
# - The `Planet Candidate` holds informaiton on the TOI data collected by [SPOC] (eg transit period, etc)
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

# -

# Some of the TOIs parameters stored on ExoFOP:

# + pycharm={"name": "#%%\n"} tags=["exe"]
tic_entry.display()

# -

# Plot of the lightcurve:


# + pycharm={"name": "#%%\n"} tags=["exe"]
plot_lightcurve(tic_entry)


# -
# ## Fitting stellar parameters
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
# * $t0_i$ time of first transit for each planet (reference time),
# * $tmax_i$ time of the last transit observed by TESS for each planet (a second reference time),
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
# Now that we have defined our transit model, we can implement it in python:
#
# [Foreman-Mackey et al 2017]: https://arxiv.org/pdf/1703.09710.pdf
# [Kipping 2013]: https://arxiv.org/abs/1308.0009
# [SHOTterm]: https://celerite2.readthedocs.io/en/latest/api/python/?highlight=SHOTerm#celerite2.terms.SHOTerm

# + pycharm={"name": "#%%\n"} tags=["def"]
def build_planet_transit_model(tic_entry):
    t = tic_entry.lightcurve.time
    y = tic_entry.lightcurve.flux
    yerr = tic_entry.lightcurve.flux_err

    n = tic_entry.planet_count
    t0s = np.array([planet.t0 for planet in tic_entry.candidates])
    depths = np.array([planet.depth for planet in tic_entry.candidates])
    periods = np.array([planet.period for planet in tic_entry.candidates])
    tmaxs = np.array([planet.tmax for planet in tic_entry.candidates])
    durations = np.array([planet.duration for planet in tic_entry.candidates])
    max_duration, min_duration = durations.max(), durations.min()

    with pm.Model() as my_planet_transit_model:
        ## define planet parameters

        # 1) d: transit duration (duration of eclipse)
        d_priors = pm.Lognormal("d", mu=np.log(0.1), sigma=10.0, shape=n)

        # 2) r: radius ratio (planet radius / star radius)
        r_priors = pm.Lognormal(
            "r", mu=0.5 * np.log(depths * 1e-3), sd=1.0, shape=n
        )
        # 3) b: impact parameter
        b_priors = xo.distributions.ImpactParameter("b", ror=r_priors, shape=n)
        planet_priors = [r_priors, d_priors, b_priors]

        ## define orbit-timing parameters

        # 1) t0: the time of the first transit in data (a reference time)
        t0_norm = pm.Bound(
            pm.Normal, lower=t0s - max_duration, upper=t0s + max_duration
        )
        t0_priors = t0_norm("t0", mu=t0s, sd=1.0, shape=n)

        # 2) period: the planets' orbital period
        p_params, p_priors_list, tmax_priors_list = [], [], []
        for n, planet in enumerate(tic_entry.candidates):
            # if only one transit in data we use the period
            if planet.has_data_only_for_single_transit:
                p_prior = pm.Pareto(
                    f"p_{planet.index}",
                    m=planet.period_min,
                    alpha=2.0 / 3.0,
                    testval=planet.period,
                )
                p_param = p_prior
                tmax_prior = planet.t0
            # if more than one transit in data we use a second time reference (tmax)
            else:
                tmax_norm = pm.Bound(
                    pm.Normal,
                    lower=planet.tmax - max_duration,
                    upper=planet.tmax + max_duration,
                )
                tmax_prior = tmax_norm(
                    f"tmax_{planet.index}",
                    mu=planet.tmax,
                    sigma=0.5 * planet.duration,
                    testval=planet.tmax,
                )
                p_prior = (tmax_prior - t0_priors[n]) / planet.num_periods
                p_param = tmax_prior

            p_params.append(p_param)  # the param needed to calculate p
            p_priors_list.append(p_prior)
            tmax_priors_list.append(tmax_prior)

        p_priors = pm.Deterministic("p", tt.stack(p_priors_list))
        tmax_priors = pm.Deterministic("tmax", tt.stack(tmax_priors_list))

        ## define stellar parameters

        # 1) f0: the mean flux from the star
        f0_prior = pm.Normal("f0", mu=0.0, sd=10.0)

        # 2) u1, u2: limb darkening parameters
        u_prior = xo.distributions.QuadLimbDark("u")
        stellar_priors = [f0_prior, u_prior]

        ## define k(t, t1; parameters)
        jitter_prior = pm.InverseGamma(
            "jitter", **pmx.estimate_inverse_gamma_parameters(1.0, 5.0)
        )
        sigma_prior = pm.InverseGamma(
            "sigma", **pmx.estimate_inverse_gamma_parameters(1.0, 5.0)
        )
        rho_prior = pm.InverseGamma(
            "rho", **pmx.estimate_inverse_gamma_parameters(0.5, 10.0)
        )
        kernel = terms.SHOTerm(sigma=sigma_prior, rho=rho_prior, Q=0.3)
        noise_priors = [jitter_prior, sigma_prior, rho_prior]

        ## define the lightcurve model mu(t;paramters)
        orbit = xo.orbits.KeplerianOrbit(
            period=p_priors,
            t0=t0_priors,
            b=b_priors,
            duration=d_priors,
            ror=r_priors,
        )
        star = xo.LimbDarkLightCurve(u_prior)
        lightcurve_models = star.get_light_curve(orbit=orbit, r=r_priors, t=t)
        lightcurve = 1e3 * pm.math.sum(lightcurve_models, axis=-1) + f0_prior
        lightcurve_models = pm.Deterministic("lightcurves", lightcurve_models)
        rho_circ = pm.Deterministic("rho_circ", orbit.rho_star)

        # Finally the GP likelihood
        residual = y - lightcurve
        gp = GaussianProcess(
            kernel, t=t, diag=yerr ** 2 + jitter_prior ** 2, mean=lightcurve
        )
        gp.marginal("obs", observed=y)

        # cache params
        my_params = dict(
            planet_params=planet_priors,
            noise_params=noise_priors,
            stellar_params=stellar_priors,
            period_params=p_params,
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
    model, planet_params, noise_params, stellar_params, period_params
):
    """Get params with maximimal log prob for sampling starting point"""
    logger.info("Optimizing sampling starting point")
    with model:
        theta = model.test_point
        kwargs = dict(start=theta, verbose=False, progress=False)
        theta = pmx.optimize(**kwargs, vars=[noise_params[0]])
        theta = pmx.optimize(**kwargs, vars=planet_params)
        theta = pmx.optimize(**kwargs, vars=noise_params)
        theta = pmx.optimize(**kwargs, vars=stellar_params)
        theta = pmx.optimize(**kwargs, vars=period_params)
    logger.info("Optimization complete!")
    return theta


# + tags=["exe"]
init_params = get_optimized_init_params(planet_transit_model, **params)
# -

# Now we can plot our initial model:

# + pycharm={"name": "#%%\n"} tags=["exe"]
model_lightcurves = [
    init_params["lightcurves"][:, i] * 1e3
    for i in range(tic_entry.planet_count)
]
plot_lightcurve(tic_entry, model_lightcurves)

# + tags=["exe"]
plot_folded_lightcurve(tic_entry, model_lightcurves)


# -

#
# ### Sampling
# With the model and priors defined, we can begin sampling

# + pycharm={"name": "#%%\n"} tags=["def"]
def start_model_sampling(model) -> MultiTrace:
    np.random.seed(TOI_NUMBER)
    with model:
        samples_trace = pmx.sample(
            tune=2000, draws=2000, chains=2, cores=1, start=init_params
        )
        return samples_trace


# + pycharm={"name": "#%%\n"} tags=["exe"]
trace = start_model_sampling(planet_transit_model)

# -

# Lets save the posteriors and sampling metadata for future use, and take a look at summary statistics

# + pycharm={"name": "#%%\n"} tags=["exe"]
tic_entry.save_data(trace=trace)
tic_entry.inference_data.get_summary_dataframe()

# + tags=["exe"]
tic_entry.inference_data.trace

# -
# ## Results
# Below are plots of the posterior probability distributuions:

# + pycharm={"name": "#%%\n"} tags=["exe"]
plot_posteriors(tic_entry)

# -
# We can also plot the best-fitting light-curve model

# + pycharm={"name": "#%%\n"} tags=["exe"]
plot_phase(tic_entry)

# -

# ### Post-processing: Eccentricity
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
star.display()

# + pycharm={"name": "#%%\n"} tags=["exe"]
if star.density_data_present:
    ecc_samples = calculate_eccentricity_weights(star, tic_entry, trace)
    ecc_samples.to_csv(
        os.path.join(tic_entry.outdir, "eccentricity_samples.csv")
    )
    plot_eccentricity_posteriors(tic_entry, ecc_samples)
else:
    logger.info(
        "Stellar data not present for TIC. Skipping eccentricity calculations."
    )

# -

# ## Citations
#

# + pycharm={"name": "#%%\n"} tags=["exe"]
with planet_transit_model:
    txt, bib = xo.citations.get_citations_for_model()
print(txt)

# + pycharm={"name": "#%%\n"} tags=["exe", "output_scroll"]
print("\n".join(bib.splitlines()) + "\n...")
# -

# ### Packages used:
#

# + pycharm={"name": "#%%\n"} tags=["exe", "output_scroll"]
import pkg_resources

dists = [str(d).replace(" ", "==") for d in pkg_resources.working_set]
for i in dists:
    print(i)
