import logging
from typing import List, Optional, Tuple, Union

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

from ..logger import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)


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
    model: Model,
    samples: List[List[float]],
    target: Union[TensorVariable, List[TensorVariable]],
    verbose: Optional[bool] = False,
) -> Union[np.ndarray, Tuple[np.ndarray]]:
    """Computes value for a model variable.

    :param Model: The pymc3 model object
    :param List[List[float]] samples: the samples to evaluate the model at
    :param InferenceData idata: The inference data trace
    :param TensorVariable target: The tensor (or list of tensors) that you want to compute for the samples
    :param Union[np.ndarray, Tuple[np.ndarray]] return_arr: The (pre-allocated) array to store the results.

    :return: np.ndarray: i rows of predictions, each with j entries.
    """

    # preprocessing
    if not isinstance(target, list):
        target = [target]
    num_target, num_samp = len(target), len(samples)
    assert num_samp > 0, f"At least 1 sample needed"

    # compile a function to compute the target
    varnames = get_untransformed_varnames(model)
    model_vars = [model[n] for n in varnames]
    func = theano.function(
        inputs=model_vars, outputs=target, on_unused_input="ignore"
    )

    # pre-generate array for data
    res = func(*samples[0])
    out_array = [
        np.zeros((len(samples), *res[i].shape)) for i in range(num_target)
    ]

    # call model function and pre-comute out
    for i in range(num_samp):
        res = func(*samples[i])
        for t in range(num_target):
            out_array[t][i, :] = res[t][:]

    if num_target == 1:
        return out_array[0]
    else:
        return out_array
