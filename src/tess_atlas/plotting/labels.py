# PLOT FILE NAMES
LIGHTCURVE_PLOT = "flux_vs_time.png"
PHASE_PLOT = "phase_plot.png"
POSTERIOR_PLOT = "posteriors.png"
ECCENTRICITY_PLOT = "eccentricity_posteriors.png"
PRIOR_PLOT = "priors.png"
DIAGNOSTIC_LIGHTCURVE_PLOT = "diagnostic_flux_vs_time.png"
DIAGNOSTIC_TRACE_PLOT = "diagnostic_trace_plot.png"
DIAGNOSTIC_RAW_LC_PLOT = "diagnostic_raw_lc_plot.png"


# PARAM CONSTANTS
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

# PLOT AXES LABELS
TIME_LABEL = "Time [days]"
TIME_SINCE_TRANSIT_LABEL = "Time since transit [days]"
FLUX_LABEL = "Relative Flux [ppt]"


# PARAM MATH LATEX LABELS
LATEX = {
    IMPACT_PARAM: r"$b$",
    MEAN_FLUX: r"$f_0$",
    ORBITAL_PERIOD: r"$P [{\rm days}]$",
    TIME_START: r"$t_{\rm min}$ [BJD]",
    TIME_END: r"$t_{\rm max}$ [BJD]",
    f"{LIMB_DARKENING_PARAM}_1": r"$u_1$",
    f"{LIMB_DARKENING_PARAM}_2": r"$u_2$",
    DURATION: r"$\tau\, [{\rm days}]$",
    RADIUS_RATIO: r"$R_{\rm p}/ R_{\star}$",
    LC_JITTER: r"$\rm{jitter}$",
    GP_SIGMA: r"$\rm{GP} \sigma$",
    GP_RHO: r"$\rm{GP} \rho$",
    RHO_CIRC: r"$\rho_{\rm circ}$",
    f"log_{DURATION}": r"$\log \rm{duration [log days]}$",
    f"log_{RADIUS_RATIO}": r"$\log (R_{\rm p}/ R_{\ast})$",
    f"log_{LC_JITTER}": r"$\log \rm{jitter}$",
    f"log_{GP_SIGMA}": r"$\log \rm{GP} \sigma$",
    f"log_{GP_RHO}": r"$\log \rm{GP} \rho$",
    f"log_{RHO_CIRC}": r"$\log \rho_{\rm circ}$",
    "eccentricity": r"$e$",
    "e": r"$e$",
    "omega": r"$\omega$",
}

PARAMS_CATEGORIES = {
    "PLANET PARAMS": [f"log_{RADIUS_RATIO}", IMPACT_PARAM],
    "PERIOD PARAMS": [TIME_START, TIME_END, ORBITAL_PERIOD, f"log_{DURATION}"],
    "STELLAR PARAMS": [
        MEAN_FLUX,
        f"{LIMB_DARKENING_PARAM}_1",
        f"{LIMB_DARKENING_PARAM}_2",
    ],
    "NOISE PARAMS": [f"log_{LC_JITTER}", f"log_{GP_SIGMA}", f"log_{GP_RHO}"],
}
