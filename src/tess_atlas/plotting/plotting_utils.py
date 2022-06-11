from typing import Dict, List, Optional, Tuple

import logging
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import math
import re

from matplotlib.ticker import ScalarFormatter

from tess_atlas.data.inference_data_tools import (
    get_posterior_samples,
    get_samples_dataframe,
    get_median_sample,
)

from ..analysis import compute_variable, get_untransformed_varnames

from tess_atlas.utils import NOTEBOOK_LOGGER_NAME

logger = logging.getLogger(NOTEBOOK_LOGGER_NAME)


UNREADABLE_MINUS = str("\u2212")
READABLE_MINUS = str("\u002D")


def get_colors(
    num_colors: int, alpha: Optional[float] = 1
) -> List[List[float]]:
    """Get a list of colorblind colors,
    :param num_colors: Number of colors.
    :param alpha: The transparency
    :return: List of colors. Each color is a list of [r, g, b, alpha].
    """
    cs = sns.color_palette(palette="colorblind", n_colors=num_colors)
    cs = [list(c) for c in cs]
    for i in range(len(cs)):
        cs[i].append(alpha)
    return cs


def format_prior_samples_and_initial_params(
    prior_samples: Dict, init_params: Dict
) -> Tuple[pd.DataFrame, Dict]:
    init_params = init_params.copy()

    # get params to log
    param_to_log = [k.split("_")[0] for k in init_params.keys() if "log" in k]
    param_to_log.append("rho_circ")

    # reduce inital param dict to only include above params
    init_params["u_1"] = init_params["u"][0]
    init_params["u_2"] = init_params["u"][1]
    init_params = {n: init_params[n] for n in prior_samples.keys()}

    prior_samples = pd.DataFrame(prior_samples)

    # log params
    for param in param_to_log:
        init_params[f"log_{param}"] = np.log(init_params[param])
        prior_samples[f"log_{param}"] = np.log(prior_samples[param])
        init_params.pop(param)
        prior_samples.drop([param], axis=1, inplace=True)

    # drop nans (from logging rho_circ)
    prior_samples.dropna(inplace=True)

    return prior_samples, init_params


def format_label_string_with_offset(ax, axis="both"):
    """Format the label string with the exponent from the ScalarFormatter"""
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis=axis, style="sci", scilimits=[-4, 4])

    axes_instances = []
    if axis in ["x", "both"]:
        axes_instances.append(ax.xaxis)
    if axis in ["y", "both"]:
        axes_instances.append(ax.yaxis)

    for ax in axes_instances:
        ax.major.formatter._useMathText = False
        ax.major.formatter._useOffset = True

        plt.draw()  # Update the text
        offset_text = ax.get_offset_text().get_text()
        label = ax.get_label().get_text()
        ax.offsetText.set_visible(False)
        ax.set_label_text(update_label(label, offset_text))


def parse_matplotlib_sf(v: str) -> float:

    if UNREADABLE_MINUS in v:
        v = v.replace(UNREADABLE_MINUS, READABLE_MINUS)

    match_number = re.compile("-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?")
    vals = [float(x) for x in re.findall(match_number, v)]

    offset = vals[-1]
    multiplyer = vals[0] if len(vals) > 1 else ""
    # sign = '-' if offset < 0 else "+"

    if multiplyer:
        multiplyer = f"{multiplyer:.0e}x "

    if offset > 100:
        offset = f"{int(offset):+}"
    else:
        offset = f"{offset:+.2f}"

    return offset, multiplyer


def update_label(old_label, offset_text):
    if offset_text == "":
        return old_label

    try:
        units = old_label[old_label.index("[") + 1 : old_label.rindex("]")]
    except ValueError:
        units = ""
    label = old_label.replace("[{}]".format(units), "")

    offset, multiplyer = parse_matplotlib_sf(offset_text)

    return f"{multiplyer}{label} [{offset} {units}]"


def get_one_dimensional_median_and_error_bar(
    data_array, fmt=".2f", quantiles=(0.16, 0.84)
) -> str:
    """Calculate the median and error bar for a given key

    Parameters
    ==========
    data_array: np.1d array
        The parameter array for which to calculate the median and error bar
    fmt: str, ('.2f')
        A format string
    quantiles: list, tuple
        A length-2 tuple of the lower and upper-quantiles to calculate
        the errors bars for.

    Returns
    =======
    str
        A latex str with {median}_{lower quant}^{upper quant}

    """

    if len(quantiles) != 2:
        raise ValueError("quantiles must be of length 2")

    quants_to_compute = np.array([quantiles[0], 0.5, quantiles[1]])
    quants = np.percentile(data_array, quants_to_compute * 100)
    median = quants[1]
    plus = quants[2] - median
    minus = median - quants[0]

    fmt = "{{0:{0}}}".format(fmt).format
    tmplate = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
    return tmplate.format(fmt(median), fmt(minus), fmt(plus))


def get_range(data, params: List[str]) -> List[List[int]]:
    """Gets bouds of dataset"""
    if isinstance(data, pd.DataFrame):
        return [[data[p].min(), data[p].max()] for p in params]
    elif isinstance(data, az.InferenceData):
        data_array = convert_to_numpy_list(data, params)
        return [[min(d), max(d)] for d in data_array]
    else:
        raise TypeError("Unexpected type provided to get_range")


def convert_to_numpy_list(
    inference_data: az.InferenceData, params: List[str]
) -> np.ndarray:
    """Converts from az.InferenceData --> 2D np.ndarray

    List[posterior_param_list_1, posterior_param_list_2...]
    each item in list is a list for the specific param
    """
    dataset = convert_to_dataset(inference_data, group="posterior")
    var_names = _var_names(params, dataset)
    plotters = list(
        xarray_var_iter(
            get_coords(dataset, {}), var_names=var_names, combined=True
        )
    )
    return np.stack([x[-1].flatten() for x in plotters], axis=0)


def get_longest_unbroken_section_of_data(t, min_break_len=10):
    """Gets longest chain of data without a break of longer than min_break_len"""
    td = np.array([t2 - t1 for t2, t1 in zip(t[1:], t)])
    t_split = np.split(t, np.where(td >= min_break_len)[0])
    split_lens = [len(ts) for ts in t_split]
    longest_t = t_split[split_lens.index(max(split_lens))][1:-1]
    idx = np.searchsorted(t, longest_t)
    return idx, longest_t


def get_lc_and_gp_from_inference_object(model, inference_data, n=1000):
    f0 = np.median(get_samples_dataframe(inference_data)[f"f0"])
    varnames = get_untransformed_varnames(model)
    samples = get_posterior_samples(
        inference_data=inference_data, varnames=varnames, size=n
    )
    median_sample = get_median_sample(
        inference_data=inference_data, varnames=varnames
    )
    lcs = compute_variable(model, samples, target=model.lightcurve_models)
    gp_mu = compute_variable(
        model, [median_sample], target=model.gp_mu, verbose=False
    )
    lcs = lcs * 1e3  # scale the lcs
    gp_model = gp_mu[0] + f0
    return lcs, gp_model


def exception_catcher(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Skipping {func}({args}, {kwargs}): {e}")
            pass

    return wrapper
