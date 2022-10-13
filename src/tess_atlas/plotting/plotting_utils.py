import logging
import math
import re
from typing import Dict, List, Optional, Tuple

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator, NullLocator, ScalarFormatter
from scipy.interpolate import interp1d

from tess_atlas.data.inference_data_tools import (
    get_median_sample,
    get_posterior_samples,
    get_samples_dataframe,
)
from tess_atlas.utils import NOTEBOOK_LOGGER_NAME

from ..analysis import compute_variable, get_untransformed_varnames

logger = logging.getLogger(NOTEBOOK_LOGGER_NAME)


UNREADABLE_MINUS = str("\u2212")
READABLE_MINUS = str("\u002D")
BACKSLASH = str("\u005c")


def fold_lightcurve_models(lc, lc_models, t0s, periods, plt_min, plt_max):
    num_lc = len(lc_models)
    xs, ys = [], []

    # fold lightcurve models
    for i in range(num_lc):
        x, fold_idx = fold_data(lc, t0s[i], periods[i], plt_min, plt_max)
        xs.append(x[fold_idx])
        ys.append(lc_models[i, fold_idx])

    # determine which lightcurve model is the longest
    longest_x_id = np.argmax([len(x) for x in xs])
    longest_x = xs[longest_x_id]

    # interpolate each lightcurve model to have same lengths
    for i, (x, y) in enumerate(zip(xs, ys)):
        f = interp1d(x, y, fill_value="extrapolate")
        ys[i] = f(longest_x)
    return longest_x, np.array(ys)


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
    ax.ticklabel_format(axis=axis, style="sci", scilimits=[-2, 2])

    axes_instances, axis_type = [], []
    if axis in ["x", "both"]:
        axis_type.append("x")
        axes_instances.append(ax.xaxis)
    if axis in ["y", "both"]:
        axis_type.append("y")
        axes_instances.append(ax.yaxis)

    for axi, axt in zip(axes_instances, axis_type):
        axi.major.formatter._useMathText = False
        axi.major.formatter._useOffset = True

        fs = rcParams["ytick.labelsize"]
        plt.draw()  # Update the text
        offset_text = axi.get_offset_text().get_text()
        label = axi.get_label().get_text()
        new_label, new_offset = update_label(label, offset_text)
        axi.offsetText.set_visible(False)
        kwg = dict(
            xycoords="axes fraction", annotation_clip=False, fontsize=fs
        )
        if axt == "x":
            ax.annotate(new_offset, xy=(1, 0), ha="right", va="bottom", **kwg)
        if axt == "y":
            ax.annotate(
                new_offset,
                xy=(0.03, 1),
                rotation=90,
                ha="left",
                va="top",
                **kwg,
            )
        new_label = new_label.replace(BACKSLASH + BACKSLASH, BACKSLASH)
        axi.set_label_text(new_label)


def parse_matplotlib_sf(v: str) -> Tuple[str, str]:
    """
    eg:
    $\times\mathdefault{10^{−3}}\mathdefault{+2.9790000000 \times 10^{1}}$
    --> "\times 10^{−3}", "+29.7"
    $\times\mathdefault{10^{−2}}\mathdefault{}$
    --> "\times 10^{−2}", ""
    """

    if UNREADABLE_MINUS in v:
        v = v.replace(UNREADABLE_MINUS, READABLE_MINUS)

    v = v.strip("$")
    v = v.replace(r"\times\mathdefault", r"\times")
    v = v.replace(r"}\mathdefault", "};")
    v = v.split(";")
    multiplyer = v[0]  # latex code
    offset = v[1]
    offset = offset.replace(r"\times", "*").replace("^", "**")
    offset = offset.replace(r"{", "(").replace(r"}", ")")
    offset = eval(offset)
    if isinstance(offset, float):
        if offset > 100:
            offset = f"{int(offset):+}"
        else:
            offset = f"{offset:+.2f}"
            if offset[-2] == "00":
                offset = offset[:-3]
    else:
        offset = ""

    return offset, multiplyer


def update_label(old_label, offset_text):
    if offset_text == "":
        return old_label, ""

    offset, multiplyer = parse_matplotlib_sf(offset_text)
    try:
        units = old_label[old_label.index("[") + 1 : old_label.rindex("]")]
    except ValueError:
        units = ""
    old_label = old_label.replace("$", "")
    label = old_label.replace(f"[{units}]", "")
    label = label.strip("\n")
    units = units.strip("\n")

    brac_str = ""
    if len(offset) > 0:
        brac_str = f"{offset} "
    if len(units) > 0:
        brac_str = f"{brac_str} {units}"
    brac_str = brac_str.strip()
    brac_str = brac_str.replace(" ", "\, ")
    if len(brac_str) > 0:
        brac_str = f"\, [{brac_str}]"

    new_label = f"${label}{brac_str}$"
    new_multiplyer_offset = f"${multiplyer}$"
    return new_label, new_multiplyer_offset


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
    unc = ""
    bigger = max(plus, minus)
    if (minus >= 0.01) and (plus >= 0.01):
        unc = "_{-" + f"{fmt(minus)}" + "}" + "^{+" + f"{fmt(plus)}" + "}"
    elif (minus >= 0.01) or (plus >= 0.01):
        unc = r"\pm " + fmt(bigger)
    else:
        expo = int(np.floor(np.log10(bigger)))
        unc = r"\pm 10^{" + str(expo) + "}"

    med = fmt(median)
    test_v = np.abs(float(med))
    if math.isclose(test_v, 0):
        med = "0.00"
    return f"${med}{unc}$"


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


def get_lc_and_gp_from_inference_object(
    model, inference_data, n=12, sample=None
):
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
    samples = [{k: v for k, v in zip(varnames, s)} for s in samples]
    return lcs, gp_model, samples


def generate_model_lightcurve(
    planet_transit_model,
    sample: Optional[Dict] = None,
    b: Optional[float] = None,
    dur: Optional[float] = None,
    f0: Optional[float] = None,
    jitter: Optional[float] = None,
    p: Optional[float] = None,
    r: Optional[float] = None,
    rho: Optional[float] = None,
    rho_circ: Optional[float] = None,
    sigma: Optional[float] = None,
    t0: Optional[float] = None,
    tmax: Optional[float] = None,
    u: Optional[List] = None,
):
    """Function to manually test some parameter values for the model"""
    if sample is None:
        assert b is not None, "Either pass 1 sample-dict or individual params"
        tmax_1 = tmax
        samp = [
            [b],
            [dur],
            f0,
            jitter,
            [p],
            [r],
            rho,
            [rho_circ],
            sigma,
            [t0],
            [tmax],
            tmax_1,
            u,
        ]
    else:
        model_varnames = get_untransformed_varnames(planet_transit_model)
        samp = [sample[n] for n in model_varnames]

    lc = compute_variable(
        model=planet_transit_model,
        samples=[samp],
        target=planet_transit_model.lightcurve_models,
    )
    return np.array(lc * 1e3)


def exception_catcher(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Skipping {func}({args}, {kwargs}): {e}")
            pass

    return wrapper


def fold_data(lc, t0, p, plt_min, plt_max):
    # wrap data around period with t0 offset
    x_fold = lc.timefold(t0=t0, p=p)
    # ignore data far away from period
    idx = np.where((plt_min <= x_fold) & (x_fold <= plt_max))
    inds = np.argsort(x_fold)
    inds = inds[np.abs(x_fold)[inds] < plt_max]
    return x_fold, inds
