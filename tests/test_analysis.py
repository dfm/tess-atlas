import pandas as pd
import pytest

import tess_atlas


@pytest.fixture()
def model():
    # this is a mock object that can has one attribute:
    # model.test_point
    # which is a dictionary of parameters
    # and one method:
    # model.check_test_point(point)
    # which returns a dataframe of the log probability of the test point
    # finally allow the model to be used as a context manager
    # with model as m:
    #     m.check_test_point(point)

    class MockModel:
        def __init__(self):
            self.test_point = dict(a=1, b=2, c=3)

        def check_test_point(self, point):
            return pd.DataFrame([0], columns=["logp"])

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    return MockModel()


def test_tess_atlas_model_checker(model):
    from tess_atlas.data.inference_data_tools import test_model

    df = test_model(model, show_summary=True)
    assert isinstance(df, pd.DataFrame)


def test_imports_are_valid():
    """Test that all imports are valid (I was getting some circular import errors)"""
    from tess_atlas.analysis.eccenticity_reweighting import (
        calculate_eccentricity_weights,
    )
    from tess_atlas.logger import get_notebook_logger
    from tess_atlas.plotting import plot_diagnostics

    assert plot_diagnostics is not None
    assert get_notebook_logger() is not None
    assert calculate_eccentricity_weights is not None
