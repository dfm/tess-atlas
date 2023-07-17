import aesara_theano_fallback.tensor as tt
import exoplanet as xo
import numpy as np
import pymc3 as pm
import pymc3_ext as pmx
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
        gp_kwargs = dict(diag=yerr**2 + jitter_prior**2, quiet=True)
        gp = GaussianProcess(kernel, t, **gp_kwargs)
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
