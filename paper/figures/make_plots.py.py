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
    major=7, minor=3, linewidth=1.5, grid=False, topon=True, righton=True
):
    rcParams["font.size"] = 30
    rcParams["font.family"] = "serif"
    rcParams["font.weight"] = "normal"
    rcParams["font.sans-serif"] = ["Computer Modern Sans"]
    rcParams["text.usetex"] = True
    rcParams["axes.labelsize"] = 30
    rcParams["axes.titlesize"] = 30
    rcParams["axes.labelpad"] = 10
    rcParams["axes.linewidth"] = linewidth
    rcParams["axes.edgecolor"] = "black"
    rcParams["xtick.labelsize"] = 25
    rcParams["ytick.labelsize"] = 25
    rcParams["xtick.direction"] = "in"
    rcParams["ytick.direction"] = "in"
    rcParams["xtick.major.size"] = major
    rcParams["xtick.minor.size"] = minor
    rcParams["ytick.major.size"] = major
    rcParams["ytick.minor.size"] = minor
    rcParams["xtick.minor.width"] = linewidth
    rcParams["xtick.major.width"] = linewidth
    rcParams["ytick.minor.width"] = linewidth
    rcParams["ytick.major.width"] = linewidth
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
    fig.savefig("radius_error.png", dpi=300, transparent=False, facecolor='white', bbox_inches='tight', )


def make_period_radius_comparison_plot():
    set_matplotlib_style_settings(major=12, minor=0)
    fig, ax = plt.subplots(2,1,figsize=(8,10), sharex=True)
    fig.subplots_adjust(hspace=0, wspace=0)
    plot_radius_period_scatterplot(ax[1])
    add_solar_sys_planet_radii_lines(ax[1])
    plot_radii_hist(ax[0])
    add_solar_sys_planet_radii_lines(ax[0])
    fig.savefig("radius_period_plot.png", dpi=300, transparent=False,facecolor='white',bbox_inches='tight', )
    return fig


    
fig = make_period_radius_comparison_plot()
make_radius_comparions_plot()
