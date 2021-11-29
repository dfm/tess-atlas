import logging
from typing import List, Optional, Tuple

import numpy as np
import theano
from arviz import InferenceData
from pymc3 import Model
from pymc3.distributions.distribution import draw_values
from pymc3.util import (
    get_default_varnames,
    get_untransformed_name,
    is_transformed_name,
)
from theano.tensor.var import TensorVariable
from tqdm.auto import tqdm

from tess_atlas.utils import NOTEBOOK_LOGGER_NAME

logger = logging.getLogger(NOTEBOOK_LOGGER_NAME)


def sample_prior(model: Model, size: Optional[int] = 10000):
    varnames = get_untransformed_varnames(model)
    try:
        samples = draw_values([model[v] for v in varnames], size=size)
        prior_samples = {}
        for i, label in enumerate(varnames):
            if label != "u":
                prior_samples[label] = np.hstack(samples[i])
            else:
                u_vals = np.hstack(samples[i])
                prior_samples["u_1"] = u_vals[::2]
                prior_samples["u_2"] = u_vals[1::2]

    except Exception as e:
        logger.error(f"Not sampling prior due to following error:\n{e}")
        return {}

    return prior_samples


def get_untransformed_varnames(model: Model) -> List[str]:
    vars = get_default_varnames(model.unobserved_RVs, True)
    names = list(
        sorted(
            set(
                [
                    get_untransformed_name(v.name)
                    if is_transformed_name(v.name)
                    else v.name
                    for v in vars
                ]
            )
        )
    )
    return names


def compute_variable(
    model: Model, samples: List[List[float]], target: TensorVariable
) -> np.ndarray:
    """Computes value for a model variable.

    :param Model model: The pymc3 model object
    :param InferenceData idata: The inference data trace
    :param TensorVariable target: The tensor (or list of tensors) that you want to compute for the samples
    :param int size: The number of samples to draw (leave it as None for all, but that's probably not what we want).

    :return: np.ndarray: i rows of predictions, each with j entries
    """
    varnames = get_untransformed_varnames(model)

    # compile a function to compute the target
    vars = [model[n] for n in varnames]
    func = theano.function(
        inputs=vars, outputs=target, on_unused_input="ignore"
    )

    #  run post-processing
    results = [
        func(*s) for s in tqdm(samples, desc="Computing model variable")
    ]
    return np.array(results)
