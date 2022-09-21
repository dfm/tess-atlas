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
#     display_name: tess
#     language: python
#     name: tess
# ---

# %load_ext autoreload
# %load_ext memory_profiler
# %load_ext autotime
# %load_ext jupyternotify
# %autoreload 2
# %matplotlib inline

# # Make Population Plots of Radius 

# +
import os, subprocess
import pandas as pd
import subprocess
import matplotlib.pyplot as plt
import os
from matplotlib import rcParams
import matplotlib.ticker as ticker
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
import string
import numpy as np
from matplotlib.ticker import ScalarFormatter
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt
from astropy.constants import R_sun, R_earth


CATALOG_COLORS = dict(
    confirmed="k", atlas="#8E24AA"
)

PLANET_COLORS = dict(
    Mercury="salmon", Earth="dodgerblue", Neptune="orchid", Jupiter="orange"
)



DOWNLOAD_COMMAND = '''wget "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+pscomppars&format=csv" -O "data/confirmed_planets.csv"'''


def load_summary(only_valid_rhats=True):
    """Loads the TESS Atlas summary stats for each analysed TOI (this also contains ExoFOP values for TOIs)"""
    df = pd.read_csv("data/atlas_pe_summary.csv")
    r_earth_to_sun = (R_earth / R_sun).value 
    r_earth_to_sun = R_earth.value 
    df["atlas_radius"] = df['r_mean'] * df["Stellar Radius (R_Sun)"] * 100
    if only_valid_rhats:
        df= df[df['rhat_ok']==True]
    df = df[df["p_mean"]  < 10**3]
    err_p = np.abs(df["p_mean"].values - df["Period (days)"].values) < 0.2
    
    df = df[err_p]
    return df



def load_confirmed_catalog(only_transit_exoplanets=True):
    """Loads summary stats for confirmed exoplanets
    
    pl_ratror	Ratio of Planet to Stellar Radius
    pl_bmasse	Planet Mass or Mass*sin(i) [Earth Mass]
    pl_orbper	Orbital Period [days]
    pl_eqt	Equilibrium Temperature [K]
    pl_orbsmax	Orbit Semi-Major Axis [au])
    st_mass	Stellar Mass [Solar mass]
    st_lum	Stellar Luminosity [log(Solar)]
    st_rad	Stellar Radius [Solar Radius]
    st_teff	Stellar Effective Temperature [K]
    """
    cat = "data/confirmed_planets.csv"
    if not os.path.isfile(cat):
        subprocess.run([DOWNLOAD_COMMAND], shell=True)
    c = pd.read_csv(cat)
    c = c[
        [
            "disc_year",
            "discoverymethod",
            "pl_bmasse",
            "pl_rade",
            "pl_radeerr1",
            "pl_radeerr2",
            "pl_ratror",
            "pl_ratrorerr1",
            "pl_ratrorerr2",
            "pl_orbper",
            "pl_orbpererr1",
            "pl_orbpererr2",
            "pl_orbsmax",
            "st_mass",
            "st_lum",
            "st_teff",
            "st_rad",
            "pl_eqt",
        ]
    ]
    if only_transit_exoplanets:
        c = c[c["discoverymethod"] == "Transit"]
    return c





def set_matplotlib_style_settings(
    major=7, minor=3, linewidth=1.5, grid=False, topon=True, righton=True, tdir="in", fs=30, tickmult=0.7
):
    rcParams["font.size"] = fs
    rcParams["font.family"] = "serif"
    rcParams["font.weight"] = "normal"
    rcParams["font.sans-serif"] = ["Computer Modern Sans"]
    rcParams["text.usetex"] = True
    rcParams["axes.labelsize"] = fs * 0.8
    rcParams["axes.titlesize"] = fs * 0.8
    rcParams["axes.labelpad"] = 10
    rcParams["axes.linewidth"] = linewidth
    rcParams["axes.edgecolor"] = "black"
    rcParams["xtick.labelsize"] = fs * tickmult
    rcParams["ytick.labelsize"] = fs * tickmult
    rcParams["xtick.direction"] = tdir
    rcParams["ytick.direction"] = tdir
    rcParams["xtick.major.size"] = major
    rcParams["xtick.minor.size"] = minor
    rcParams["ytick.major.size"] = major
    rcParams["ytick.minor.size"] = minor
    rcParams["xtick.minor.width"] = linewidth
    rcParams["xtick.major.width"] = linewidth
    rcParams["ytick.minor.width"] = linewidth
    rcParams["ytick.major.width"] = linewidth
    rcParams["savefig.dpi"] = 300
    rcParams["xtick.top"] = topon
    rcParams["ytick.right"] = righton
    rcParams["axes.grid"] = grid
    rcParams["axes.titlepad"] = 8
    rcParams['text.latex.preamble'] = r'\usepackage{mathabx}'





def load_planet_data():
    d = pd.read_csv("data/planet_data.csv")
    d.index = d.name
    return d




    
def plot_radius_period_scatterplot(ax):
    c = load_confirmed_catalog()
    
    ax.scatter( c.pl_rade, c.pl_orbper, s=5, color=CATALOG_COLORS['confirmed'])
    atlas = load_summary()
    ax.scatter( atlas.atlas_radius, atlas.p_mean, s=5, alpha=0.8, color=CATALOG_COLORS['atlas'])


    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: "{:,g}".format(y)))
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(
            lambda y, _: "{:,g}".format(y)
            if y < 1000
            else "$10^{" + str(int(np.log10(y))) + "}$"
        )
    )
    ax.set_ylabel("Period [days]")
    ax.set_xlabel("Radius [$R_{\oplus}$]")
    ax.set_ylim(0.1, 3000)
    ax.set_xlim(0.3, 50)
    ax.tick_params(axis='x',which='major',top=True,bottom=True)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: "{:g}".format(y)))


def add_solar_sys_planet_radii_lines(ax, label_range=[], annotate=False):
    ss_data = load_planet_data().T.to_dict()
    for k in ["Jupiter", "Neptune", "Earth",]:
        pT, pR = ss_data[k]["period"], ss_data[k]["radius"]
        ax.vlines(pR, ymin=0, ymax=10000, lw=0.2, color='black', zorder=-100)
        if len(label_range)>0:
            ax.vlines(pR, ymin=label_range[0], ymax=label_range[1], lw=1.5, color='black', zorder=0)
    

def plot_radii_hist(ax):

    c = load_confirmed_catalog()
    c = c[c.discoverymethod == "Transit"]
    ss_radii = load_planet_data()["radius"].to_dict()
    ss_radii = {k: ss_radii[k] for k in ["Earth", "Neptune", "Jupiter"]}
    ss_radii_inv = {v: k for k, v in ss_radii.items()}

    c = c[c["pl_orbper"] < 100]
    bins = np.geomspace(0.4, 30, 60)
    kwgs = dict(bins=bins, density=True, histtype="step", lw=3,)
    cts, b, _ = ax.hist(
        c.pl_rade,  color=CATALOG_COLORS['confirmed'], **kwgs, label="Verified Exoplanets"
    )

    atlas = load_summary()
    atlas = atlas[atlas['p_mean'] < 100]
    atlas = atlas[atlas["Period (days)"] < 100]
    ax.hist(
        atlas.atlas_radius,  color=CATALOG_COLORS['atlas'], **kwgs,  label="TESS Atlas"
    )

    
    ax.set_xscale("log")
    ax.set_ylabel("Density")

    tick_bot, ax_top = max(cts) * 1.1, max(cts) *1.1
    ax.set_ylim(0, ax_top)
    
    fs = 18
    ax.tick_params(bottom=False, )
    
    yr = [0, 600]

    kw = dict(
        alpha=0.15,
        lw=0,
        zorder=-10,
    )
    ka = dict(ha="left", va="bottom", xycoords="data", fontsize=fs, rotation=0)
    ytxt = max(cts) * 0.05
    ECol, NCol, JCol = (
        PLANET_COLORS["Earth"],
        PLANET_COLORS["Neptune"],
        PLANET_COLORS["Jupiter"],
    )
    ax.fill_betweenx(yr, 1, 1.75, color=ECol, **kw)
    ax.annotate("Super\nEarths", xy=(1, ytxt), **ka, color=ECol)
    ax.fill_betweenx(yr, 1.75, 3, color=JCol, **kw)
    ax.annotate(
        "Sub\nNeptune",
        xy=(1.75, ytxt),
        **ka,
        color=JCol,
    )
    
    ytxt = max(cts) * 0.65
    ka = dict(ha="right", va="bottom", xycoords="data", fontsize=20, rotation=90, color='gray')
    ax.annotate("$R_{\\rm Earth}$", xy=(1, ytxt), **ka)
    ytxt = max(cts) * 0.5
    ax.annotate("$R_{\\rm Neptune}$", xy=(3.88, ytxt), **ka)
    ax.annotate("$R_{\\rm Jupiter}$", xy=(11.13, ytxt), **ka)
    
    ax.set_xlim(0.4, 30)
    ax.set_yticks([])
    
    ax.tick_params(axis='x',which='major',top=False,bottom=True)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: "{:g}".format(y)))
    ax.legend(fontsize=18, frameon=False, loc='upper right') # loc=(0.5, 0.4),)


def scatter_comparison(ax, x1, x2,):
    x1, x2 = np.array(x1), np.array(x2)
    non_zero = np.nonzero(x2)
    x1 = x1[non_zero]
    x2 = x2[non_zero]
    ax.scatter(x1,x2, s=10, c=CATALOG_COLORS["atlas"], alpha=0.2)
    ax.axline((0, 0), (1, 1), linewidth=1.5, color='k')
    
    minv, maxv = min(min(x1), min(x2)), max(max(x1), max(x2))
    ax.set_xlim(minv , maxv)
    ax.set_ylim(minv , maxv)
    ax.set_xscale('log')
    ax.set_yscale('log')

#     ax.set_aspect('equal', adjustable='box')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: "{:g}".format(y)))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: "{:g}".format(y)))


def plot_residual_hist(ax, x1, x2, bins=np.linspace(-0.30,0.30,50), use_abs=False):
    diff = (x1/x2) -1
    if use_abs:
        diff = np.abs(diff)
    cts, _, _ = ax.hist(diff, density=True, histtype='step', bins=bins, lw=3, color=CATALOG_COLORS["atlas"])
    ax.vlines(0, ymin=0, ymax=max(cts)*1.2, color='k', lw=1.5)
    ax.set_xlim(min(bins), max(bins))
    ax.set_ylim(0, max(cts)*1.2)
    ax.set_yticks([])

    
def make_radius_comparions_plot():
    # clean data
    summary_df = load_summary()
    keep = np.abs(summary_df.r_mean - summary_df.exo_r) < 0.5
    summary_df = summary_df[keep]
    summary_df = summary_df[summary_df.atlas_radius  < 100]
    summary_df = summary_df[summary_df.p_mean  < 100]
    summary_df = summary_df[summary_df["Period (days)"]  < 100]

    fig, axes = plt.subplots(1,2, figsize=(10, 5))
    x1,  x2 = summary_df.atlas_radius, summary_df["Planet Radius (R_Earth)"]
    scatter_comparison(axes[0], x1, x2)
    plot_residual_hist(axes[1], x1, x2 , bins=np.linspace(-1,1,25))
    axes[0].set_ylabel(r'$R_{\rm ExoFOP}\ [R_{\rm \Earth}]$')
    axes[0].set_xlabel(r'$R_{\rm Atlas}\ [R_{\rm \Earth}]$')
    axes[1].set_xlabel(r'$R_{\rm Atlas}/R_{\rm ExoFOP} - 1$')
    axes[1].set_ylabel('Density')
    axes[0].set_xticks([1,10,100])
    axes[0].set_yticks([1,10,100])
    fig.tight_layout()
    fig.savefig("radius_error.png", transparent=False, facecolor='white', bbox_inches='tight', )


def make_period_radius_comparison_plot():
    set_matplotlib_style_settings(major=12, minor=0)
    fig, ax = plt.subplots(2,1,figsize=(8,10), sharex=True)
    fig.subplots_adjust(hspace=0, wspace=0)
    plot_radius_period_scatterplot(ax[1])
    add_solar_sys_planet_radii_lines(ax[1])
    plot_radii_hist(ax[0])
    add_solar_sys_planet_radii_lines(ax[0])
    fig.savefig("radius_period_plot.png", transparent=False,facecolor='white',bbox_inches='tight', )
    return fig


    
fig = make_period_radius_comparison_plot()
make_radius_comparions_plot()
# -

# # Make TOI specific plots

# ! download_toi 174 --outdir toi_174_files
# ! download_toi 103 --outdir toi_174_files

# +
from tess_atlas.data import TICEntry
import os
import aesara_theano_fallback.tensor as tt
import exoplanet as xo
import numpy as np
import pandas as pd
import pymc3 as pm
import pymc3_ext as pmx
from arviz import InferenceData
from celerite2.theano import GaussianProcess, terms




DEPTH = "depth"
DURATION = "dur"
RADIUS_RATIO = "r"
TIME_START = "tmin"
TIME_END = "tmax"
ORBITAL_PERIOD = "p"
MEAN_FLUX = "f0"
LC_JITTER = "jitter"
GP_RHO = "rho"
GP_SIGMA = "sigma"
RHO_CIRC = "rho_circ"  # stellar density at e=0
LIMB_DARKENING_PARAM = "u"
IMPACT_PARAM = "b"


def get_test_duration(min_durations, max_durations, durations):
    largest_min_duration = np.amax(
        np.array([durations, 2 * min_durations]), axis=0
    )
    smallest_max_duration = np.amin(
        np.array([largest_min_duration, 0.99 * max_durations]), axis=0
    )
    return smallest_max_duration


def build_planet_transit_model(tic_entry):
    t = tic_entry.lightcurve.time
    y = tic_entry.lightcurve.flux
    yerr = tic_entry.lightcurve.flux_err

    n = tic_entry.planet_count
    tmins = np.array([planet.tmin for planet in tic_entry.candidates])
    depths = np.array([planet.depth for planet in tic_entry.candidates])
    durations = np.array([planet.duration for planet in tic_entry.candidates])
    max_durations = np.array(
        [planet.duration_max for planet in tic_entry.candidates]
    )
    min_durations = np.array(
        [planet.duration_min for planet in tic_entry.candidates]
    )
    test_duration = get_test_duration(min_durations, max_durations, durations)

    with pm.Model() as my_planet_transit_model:
        ## define planet parameters

        # 1) d: transit duration (duration of eclipse)
        d_priors = pm.Bound(
            pm.Lognormal, lower=min_durations, upper=max_durations
        )(
            name=DURATION,
            mu=np.log(durations),
            sigma=np.log(1.2),
            shape=n,
            testval=test_duration,
        )

        # 2) r: radius ratio (planet radius / star radius)
        r_priors = pm.Lognormal(
            name=RADIUS_RATIO, mu=0.5 * np.log(depths * 1e-3), sd=1.0, shape=n
        )
        # 3) b: impact parameter
        b_priors = xo.distributions.ImpactParameter(
            name=IMPACT_PARAM, ror=r_priors, shape=n
        )
        planet_priors = [r_priors, d_priors, b_priors]

        ## define orbit-timing parameters

        # 1) tmin: the time of the first transit in data (a reference time)
        tmin_norm = pm.Bound(
            pm.Normal, lower=tmins - max_durations, upper=tmins + max_durations
        )
        tmin_priors = tmin_norm(
            TIME_START, mu=tmins, sigma=0.5 * durations, shape=n, testval=tmins
        )

        # 2) period: the planets' orbital period
        p_params, p_priors_list, tmax_priors_list = [], [], []
        for n, planet in enumerate(tic_entry.candidates):
            # if only one transit in data we use the period
            if planet.has_data_only_for_single_transit:
                p_prior = pm.Pareto(
                    name=f"{ORBITAL_PERIOD}_{planet.index}",
                    m=planet.period_min,
                    alpha=2.0 / 3.0,
                    testval=planet.period,
                )
                p_param = p_prior
                tmax_prior = planet.tmin
            # if more than one transit in data we use a second time reference (tmax)
            else:
                tmax_norm = pm.Bound(
                    pm.Normal,
                    lower=planet.tmax - planet.duration_max,
                    upper=planet.tmax + planet.duration_max,
                )
                tmax_prior = tmax_norm(
                    name=f"{TIME_END}_{planet.index}",
                    mu=planet.tmax,
                    sigma=0.5 * planet.duration,
                    testval=planet.tmax,
                )
                p_prior = (tmax_prior - tmin_priors[n]) / planet.num_periods
                p_param = tmax_prior

            p_params.append(p_param)  # the param needed to calculate p
            p_priors_list.append(p_prior)
            tmax_priors_list.append(tmax_prior)

        p_priors = pm.Deterministic(ORBITAL_PERIOD, tt.stack(p_priors_list))
        tmax_priors = pm.Deterministic(TIME_END, tt.stack(tmax_priors_list))

        ## define stellar parameters

        # 1) f0: the mean flux from the star
        f0_prior = pm.Normal(name=MEAN_FLUX, mu=0.0, sd=10.0)

        # 2) u1, u2: limb darkening parameters
        u_prior = xo.distributions.QuadLimbDark("u")
        stellar_priors = [f0_prior, u_prior]

        ## define k(t, t1; parameters)
        jitter_prior = pm.InverseGamma(
            name=LC_JITTER, **pmx.estimate_inverse_gamma_parameters(1.0, 5.0)
        )
        sigma_prior = pm.InverseGamma(
            name=GP_SIGMA, **pmx.estimate_inverse_gamma_parameters(1.0, 5.0)
        )
        rho_prior = pm.InverseGamma(
            name=GP_RHO, **pmx.estimate_inverse_gamma_parameters(0.5, 10.0)
        )
        kernel = terms.SHOTerm(sigma=sigma_prior, rho=rho_prior, Q=0.3)
        noise_priors = [jitter_prior, sigma_prior, rho_prior]

        ## define the lightcurve model mu(t;paramters)
        orbit = xo.orbits.KeplerianOrbit(
            period=p_priors,
            t0=tmin_priors,
            b=b_priors,
            duration=d_priors,
            ror=r_priors,
        )
        star = xo.LimbDarkLightCurve(u_prior)
        lightcurve_models = star.get_light_curve(orbit=orbit, r=r_priors, t=t)
        lightcurve = 1e3 * pm.math.sum(lightcurve_models, axis=-1) + f0_prior
        my_planet_transit_model.lightcurve_models = lightcurve_models
        rho_circ = pm.Deterministic(name=RHO_CIRC, var=orbit.rho_star)

        # Finally the GP likelihood
        residual = y - lightcurve
        gp = GaussianProcess(
            kernel, t=t, diag=yerr**2 + jitter_prior**2, quiet=True
        )
        gp.marginal(name="obs", observed=residual)
        my_planet_transit_model.gp_mu = gp.predict(residual, return_var=False)

        # cache params
        my_params = dict(
            planet_params=planet_priors,
            noise_params=noise_priors,
            stellar_params=stellar_priors,
            period_params=p_params,
        )
    return my_planet_transit_model, my_params

tic_entry = TICEntry.from_cache(174, outdir="toi_174_files")
tic_entry.lightcurve.filter_non_transit_data(tic_entry.candidates)
planet_transit_model, _  = build_planet_transit_model(tic_entry)

# +
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tess_atlas.plotting.phase_plotter import add_phase_data_to_ax, _preprocess_phase_plot_data


def plot_toi_phase_subplots(tic_entry, planet_transit_model):
    initial_params = tic_entry.optimized_params.to_dict()
    inference_data = tic_entry.inference_data
    model = planet_transit_model
    kwgs = _preprocess_phase_plot_data(tic_entry, model, inference_data, initial_params, kwgs={"num_lc":1000}) 
    fs = 20
    set_matplotlib_style_settings(fs=20, tickmult=0.8)
    labelfs = rcParams
    n = tic_entry.planet_count
    fig, axes = plt.subplots(n,1, figsize=(6.5, 3*n), sharex=True)
    kwgs.update(dict(
        save=False, legend=2, annotate_with_period=False, default_fs=20, legend_fs = 16
    ))
    for i in range(tic_entry.planet_count):
        ax = axes[i]
        add_phase_data_to_ax(ax, i, tic_entry, **kwgs)

    for ax in axes:
        ax.yaxis.set_major_locator(ticker.MaxNLocator(2))
        ax.set_xlim(-.2,.2)
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.1f}'))
        ax.set_ylabel("Flux [ppt]")
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.savefig("toi_174_phase.png", transparent=False,facecolor='white',bbox_inches='tight')


plot_toi_phase_subplots(tic_entry, planet_transit_model)




# +
from tess_atlas.plotting.corner_plotter import plot_posteriors

set_matplotlib_style_settings(major=2, tdir='out', tickmult=0.6)

plot_posteriors(
    tic_entry, 
    tic_entry.inference_data, 
    tic_entry.optimized_params.to_dict(), 
    plot_params=["r", "b", "p", "dur"],
    title=False,
    fname="planet_{}_posterior.png",
)



# ! rm planet_1_posterior.png planet_2_posterior.png planet_4_posterior.png planet_5_posterior.png


# +
import pandas as pd 
from tess_atlas.plotting.corner_plotter import plot_eccentricity_posteriors

set_matplotlib_style_settings(major=2, tdir='out', tickmult=0.6)

ecc_samples = pd.read_csv("toi_174_files/eccentricity_samples.csv")
plot_eccentricity_posteriors(tic_entry, ecc_samples, title=False, fname="ecc_{}_posterior.png",)

# ! rm ecc_1_posterior.png  ecc_2_posterior.png  ecc_4_posterior.png  ecc_5_posterior.png
# -

tic_entry_2 = TICEntry.from_cache(103, outdir="toi_103_files")
tic_entry_2.lightcurve.filter_non_transit_data(tic_entry_2.candidates)
planet_transit_model_2, _  = build_planet_transit_model(tic_entry_2)


# +

def plot_lightcurve_gp_and_residuals(
    tic_entry, model,  num_lc=12
):
    set_matplotlib_style_settings(fs=30, tickmult=0.8, major = 10 )
    colors = get_colors(tic_entry.planet_count)
    t = tic_entry.lightcurve.time
    y = tic_entry.lightcurve.flux
    lcs, gp_model, _ = get_lc_and_gp_from_inference_object(
        model, tic_entry.inference_data, n=num_lc
    )
    raw_lc = tic_entry.lightcurve.raw_lc
    raw_t, raw_y = raw_lc.time.value, 1e3 * (raw_lc.flux.value - 1)
    
    idx = [i for i in range(len(t))]

    fig, ax = plt.subplots(1, 1, figsize=(10, 4), sharex=True)

    ax.scatter(raw_t, raw_y, c="gray", label="Raw Data", s=1, alpha=0.5)
    transit_mask = tic_entry.lightcurve.get_transit_mask(tic_entry.candidates, 0.295)

    ax.scatter(t[idx], y[idx], c="k", label="Filtered Data", s=1)

    for i in range(tic_entry.planet_count):
        lc = np.median(lcs[..., i], axis=0)
        model_y = lc[idx] + gp_model[idx]
        m = model_y.copy()
        m[~transit_mask] = np.nan
        ax.plot(
            t[idx],
            m,
            label=f"Model",
            color=colors[i],
            lw = 5,
            alpha=0.5
        )

    l = ax.legend(loc=(1.03, 0.5), fontsize=20,  markerscale=8, frameon=False)
    ax.set_ylabel("Flux [ppt]")
    ax.set_xlabel("Time [days]")
    fig.subplots_adjust(hspace=0, wspace=0)
    plt.xlim(1340, 1347)
    plt.ylim(-13, 8)
    plt.yticks([0, -10])
    plt.tight_layout()
    


plot_lightcurve_gp_and_residuals(tic_entry_2, planet_transit_model_2)
plt.savefig("raw_data_toi_103.png")

