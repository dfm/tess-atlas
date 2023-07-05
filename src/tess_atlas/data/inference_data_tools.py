import logging
import os
from typing import List, Optional

import arviz as az
import numpy as np
import pandas as pd
import pymc3_ext as pmx

from tess_atlas.utils import NOTEBOOK_LOGGER_NAME

logger = logging.getLogger(NOTEBOOK_LOGGER_NAME)

SAMPLES_FNAME = "samples.csv"
INFERENCE_DATA_FNAME = "inference_data.netcdf"


def get_idata_fname(outdir):
    return os.path.join(outdir, INFERENCE_DATA_FNAME)


def load_inference_data(outdir: str) -> az.InferenceData:
    fname = get_idata_fname(outdir)
    if not os.path.isfile(fname):
        raise FileNotFoundError(f"{fname} not found.")
    try:
        inference_data = az.from_netcdf(fname)
        logger.info(f"Inference data loaded from {fname}")
        return inference_data
    except Exception as e:
        logger.error(f"Cant read inference file: {e}")


def save_inference_data(inference_data: az.InferenceData, outdir: str):
    fname = get_idata_fname(outdir)
    inference_data.to_netcdf(
        filename=fname, groups=["posterior", "log_likelihood", "sample_stats"]
    )
    save_samples(inference_data, outdir)


def summary(
    inference_data: az.InferenceData, just_planet_params=False
) -> pd.DataFrame:
    """Returns a dataframe with the mean+sd of each candidate's p, b, r"""
    df = az.summary(
        inference_data, var_names=["~lightcurves"], filter_vars="like"
    )
    if just_planet_params:
        df = (
            df.transpose()
            .filter(regex=r"(.*p\[.*)|(.*r\[.*)|(.*b\[.*)")
            .transpose()
        )
        df = df[["mean", "sd"]]
        df["parameter"] = df.index
        df.set_index(["parameter"], inplace=True, append=False, drop=True)

    RHAT_THRESHOLD = 1.1
    bogus_params = []
    for param, row in df.iterrows():
        if row["r_hat"] >= RHAT_THRESHOLD:
            bogus_params.append(param)

    if len(bogus_params) > 0:
        logger.warning(
            f"Sampler may not have converged! r-hat > {RHAT_THRESHOLD} for {bogus_params}"
        )

    # TODO
    """
    for each b_param in all_params:
        if median(b_param) > 0.8:
            logger.warning("b[i] > 0.8 --> may be a grazing system!")
    """

    return df


def get_samples_dataframe(inference_data: az.InferenceData) -> pd.DataFrame:
    samples = inference_data.to_dataframe(groups=["posterior"])
    # converting column labels to only include posterior label (removing group)
    new_cols = []
    for col in list(samples.columns):
        if isinstance(col, tuple):
            posterior_label, group = col
            new_cols.append(posterior_label)
        else:
            new_cols.append(col)
    samples.columns = new_cols
    return samples


def get_posterior_samples(
    inference_data: az.InferenceData,
    varnames: List[str],
    size: Optional[int] = None,
) -> List[List[float]]:
    """Flattens posterior samples (from chains) and returns List of samples"""
    flat_samps = inference_data.posterior.stack(sample=("chain", "draw"))
    total_num_samp = len(flat_samps.sample)
    if size is not None and size <= total_num_samp:
        indices = np.random.randint(len(flat_samps.sample), size=size)
    else:
        indices = np.arange(len(flat_samps.sample))
    return np.array(
        [[flat_samps[n].values[..., i] for n in varnames] for i in indices]
    )


def get_median_sample(
    inference_data: az.InferenceData, varnames: List[str]
) -> List[List[float]]:
    """Get the median sample for each param"""
    samples = get_posterior_samples(inference_data, varnames)
    param_lists = [np.stack(param_list) for param_list in samples.T]
    return [np.median(p, axis=0) for p in param_lists]


def convert_to_samples_dict(varnames: List[str], samples: np.ndarray):
    """samples obtained from get_posterior_samples"""
    samples_dict = {}
    for i, label in enumerate(varnames):
        if label != "u":
            samples_dict[label] = np.hstack(samples[..., i])
        else:
            u_vals = np.hstack(samples[..., i])
            samples_dict["u_1"] = u_vals[::2]
            samples_dict["u_2"] = u_vals[1::2]
    return samples_dict


def save_samples(inference_data, outdir):
    fpath = os.path.join(outdir, SAMPLES_FNAME)
    samples = get_samples_dataframe(inference_data)
    samples.to_csv(fpath, index=False)


def check_df_for_finites(df):
    if df.isnull().values.any():
        raise ValueError(f"The model(testval) has a nan:\n{df}")
    if np.isinf(df).values.sum() > 0:
        raise ValueError(f"The model(testval) has an inf:\n{df}")


def check_dict_for_finites(d):
    for k, v in d.items():
        if np.isnan(v).any():
            raise ValueError(f"The testval['{k}'] has a nan:\n{d}")


def test_model(model, point=None, show_summary=False):
    """Test a point in the model and assure no nans"""
    if point is None:
        point = model.test_point
    check_dict_for_finites(point)
    with model:
        test_prob = model.check_test_point(point)
        test_prob.name = "log P(test-point)"
        check_df_for_finites(test_prob)
        if show_summary:
            test_pt = pd.Series(
                {
                    k: str(round(np.array(v).flatten()[0], 3))
                    for k, v in model.test_point.items()
                },
                name="Test Point",
            )
            return pd.concat([test_pt, test_prob], axis=1)


def get_optimized_init_params(
    model,
    planet_params,
    noise_params,
    stellar_params,
    period_params,
    theta=None,
    verbose=False,
    return_all=False,
    quick=False,
):
    """Get params with maximimal log prob for sampling starting point

    planet_params:  [radius ratio, duration, impact parameter]
    noise_params: [jitter, GP sigma, GP rho]
    stellar_params: [f0, limb-darkening]
    period_params: [period for single, tmax for other]

    """
    logger.info("Optimizing sampling starting point")
    with model:
        cache = []
        if theta is None:
            theta = model.test_point
        init_logp = get_logp(model, theta)
        cache.append(dict(theta=theta, logp=init_logp))
        kwargs = dict(verbose=verbose, progress=verbose)
        all_params = [*planet_params, *noise_params, *period_params]
        timing_params = [planet_params[1], period_params[0]]
        radius_ratio = [planet_params[0]]
        jitter = [noise_params[0]]
        optimization_order = [
            jitter,
            planet_params,
            noise_params,
            stellar_params,
            period_params,
            radius_ratio,
            timing_params,
            all_params,
        ]
        if quick:
            theta = pmx.optimize(theta, period_params, **kwargs)
            cache.append(dict(theta=theta, logp=get_logp(model, theta)))
        else:
            for _ in range(2):
                for optimization_param in optimization_order:
                    theta = pmx.optimize(theta, optimization_param, **kwargs)
                    cache.append(
                        dict(theta=theta, logp=get_logp(model, theta))
                    )

        final_logp = get_logp(model, theta)
        logger.info(
            f"Optimization complete! "
            f"(logp: {init_logp:.2f} -> {cache[-1]['logp']:.2f})"
        )

        # format theta values
        for i in range(len(cache)):
            t = cache[i]["theta"]
            cache[i]["theta"] = {k: v.tolist() for k, v in t.items()}

        if return_all:
            return cache
        else:
            return cache[-1]["theta"]


def get_logp(model, point):
    with model:
        return model.check_test_point(test_point=point)["obs"]
