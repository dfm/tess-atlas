from typing import Optional, List, Tuple
import numpy as np
import theano
from pymc3.util import (
    get_default_varnames,
    get_untransformed_name,
    is_transformed_name,
)
from pymc3 import Model
from arviz import InferenceData
from tqdm.auto import tqdm

from theano.tensor.var import TensorVariable


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
