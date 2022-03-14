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


def load_inference_data(outdir: str):
    fname = get_idata_fname(outdir)
    if not os.path.isfile(fname):
        raise FileNotFoundError(f"{fname} not found.")
    try:
        inference_data = az.from_netcdf(fname)
        logger.info(f"Inference data loaded from {fname}")
        return inference_data
    except Exception as e:
        logger.error(f"Cant read inference file: {e}")


def save_inference_data(inference_data, outdir: str):
    fname = get_idata_fname(outdir)
    inference_data.to_netcdf(
        filename=fname, groups=["posterior", "log_likelihood", "sample_stats"]
    )
    save_samples(inference_data, outdir)


def summary(inference_data) -> pd.DataFrame:
    """Returns a dataframe with the mean+sd of each candidate's p, b, r"""
    df = az.summary(
        inference_data, var_names=["~lightcurves"], filter_vars="like"
    )
    df = (
        df.transpose()
        .filter(regex=r"(.*p\[.*)|(.*r\[.*)|(.*b\[.*)")
        .transpose()
    )
    df = df[["mean", "sd"]]
    df["parameter"] = df.index
    df.set_index(["parameter"], inplace=True, append=False, drop=True)
    return df


def get_samples_dataframe(inference_data) -> pd.DataFrame:
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
    inference_data, varnames: List[str], size: Optional[int] = None
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


def test_model(model, point=None, show_summary=False):
    """Test a point in the model and assure no nans"""
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
):
    """Get params with maximimal log prob for sampling starting point"""
    logger.info("Optimizing sampling starting point")
    with model:
        if theta is None:
            theta = model.test_point
        init_logp = get_logp(model, theta)
        kwargs = dict(verbose=verbose, progress=verbose)
        theta = pmx.optimize(theta, [noise_params[0]], **kwargs)
        theta = pmx.optimize(theta, planet_params, **kwargs)
        theta = pmx.optimize(theta, noise_params, **kwargs)
        theta = pmx.optimize(theta, stellar_params, **kwargs)
        theta = pmx.optimize(theta, period_params, **kwargs)
        final_logp = get_logp(model, theta)
        logger.info(
            f"Optimization complete! " f"(logp: {init_logp} -> {final_logp}"
        )
        return {k: v.tolist() for k, v in theta.items()}


def get_logp(model, point):
    with model:
        return model.check_test_point(test_point=point)["obs"]
