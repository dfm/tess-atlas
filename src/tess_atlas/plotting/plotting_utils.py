from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns


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
