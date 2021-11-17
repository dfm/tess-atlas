TIME_LABEL = "Time [days]"
TIME_SINCE_TRANSIT_LABEL = "Time since transit [days]"
FLUX_LABEL = "Relative Flux [ppt]"
LIGHTCURVE_PLOT = "flux_vs_time.png"
PHASE_PLOT = "phase_plot.png"
FOLDED_LIGHTCURVE_PLOT = "folded_flux_vs_time.png"
POSTERIOR_PLOT = "posteriors.png"
ECCENTRICITY_PLOT = "eccentricity_posteriors.png"
PRIOR_PLOT = "priors.png"

LATEX = {
    "b": r"$b$",
    "f0": r"$f_0$",
    "p": r"$p$ [days]",
    "t0": r"$t_0$ [BJD]",
    "tmax": r"$t_{\rm max}$ [BJD]",
    "u_1": r"$u_1$",
    "u_2": r"$u_2$",
    "d": r"$d$",
    "r": r"$r$",
    "jitter": r"$\rm{j}$",
    "sigma": r"$\sigma$",
    "rho": r"$\rho$",
    "rho_circ": r"$\rho_{\rm circ}$",
    "log_d": r"$\log d$",
    "log_r": r"$\log r$",
    "log_jitter": r"$\log \rm{j}$",
    "log_sigma": r"$\log \sigma$",
    "log_rho": r"$\log \rho$",
    "log_rho_circ": r"$\log \rho_{\rm circ}$",
}

PARAMS_CATEGORIES = {
    "PLANET PARAMS": ["log_r", "log_d", "b"],
    "PERIOD PARAMS": ["t0", "tmax", "p"],
    "STELLAR PARAMS": ["f0", "u_1", "u_2"],
    "NOISE PARAMS": ["log_jitter", "log_sigma", "log_rho"],
}
