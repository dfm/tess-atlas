import logging
import math
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
from matplotlib.ticker import ScalarFormatter

from ..logger import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)

UNREADABLE_MINUS = str("\u2212")
READABLE_MINUS = str("\u002D")
BACKSLASH = str("\u005c")


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
    """
    Format prior samples and initial params for plotting.
    Some formatting includes logging certain params, dropping nans, etc.

    :param prior_samples: The prior samples.
    :param init_params: The initial params.
    :return: The formatted prior samples and initial params.
    """
    init_params = init_params.copy()

    # get params that need to be converted from linear->log
    param_to_log = [k.split("_")[0] for k in init_params.keys() if "log" in k]
    param_to_log.append("rho_circ")

    # reduce inital param dict to only include above params
    init_params["u_1"] = init_params["u"][0]
    init_params["u_2"] = init_params["u"][1]
    init_params = {n: init_params[n] for n in prior_samples.keys()}

    prior_samples = pd.DataFrame(prior_samples)

    # log(params)
    for param in param_to_log:
        init_params[f"log_{param}"] = np.log(init_params[param])
        prior_samples[f"log_{param}"] = np.log(prior_samples[param])
        init_params.pop(param)
        prior_samples.drop([param], axis=1, inplace=True)

    # drop nans (from logging rho_circ)
    prior_samples.dropna(inplace=True)

    return prior_samples, init_params


def format_hist_axes_label_string_with_offset(ax, axis="both") -> None:
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
        new_label, new_offset = __update_label(label, offset_text)
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


def __parse_matplotlib_sf(v: str) -> Tuple[str, str]:
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


def __update_label(old_label, offset_text):
    if offset_text == "":
        return old_label, ""

    offset, multiplyer = __parse_matplotlib_sf(offset_text)
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


def get_longest_unbroken_section_of_data(
    t, min_break_len=10
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Get the longest chain of data without a break of longer than min_break_len
    Parameters
    ==========
    t: np.1d array
        The time array
    min_break_len: int, (10)
        The minimum length of a break in the data to be considered a break
    percent_data: int
        The percentage of data that is in the longest unbroken section

    Returns
    =======
    idx: np.ndarray
        The indices of the longest unbroken section of data
    longest_t: np.ndarray
        The time array of the longest unbroken section of datas
    percent_data: int
        The percentage of data that is in the longest unbroken section
    """
    td = np.array([t2 - t1 for t2, t1 in zip(t[1:], t)])
    t_split = np.split(t, np.where(td >= min_break_len)[0])
    split_lens = [len(ts) for ts in t_split]
    longest_t = t_split[split_lens.index(max(split_lens))][1:-1]
    idx = np.searchsorted(t, longest_t)
    percent_data = int(100 * (len(idx) / len(t)))
    return idx, longest_t, percent_data


def exception_catcher(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Skipping {func}({args}, {kwargs}): {e}")
            pass

    return wrapper
