"""Module to run unittests for functions in the template notebook"""
import os

import pandas as pd
import pymc3 as pm
import pytest
import testbook

TEMPLATE_NOTEBOOK = "tess_atlas/template.ipynb"


@pytest.fixture(scope='module')
def notebook():
    """Share kernel with the module after executing the cells with tags"""
    tags_to_execute = ["def"]
    with testbook.testbook(TEMPLATE_NOTEBOOK, execute=tags_to_execute) as notebook:
        notebook.allow_errors = True
        notebook.execute()
        yield notebook


def test_exoplanent_import_version_number(notebook):
    from packaging import version
    notebook.inject(
        """
        print(xo.__version__)
        """
    )
    version_number = notebook.cells[-1]['outputs'][0]['text'].strip()
    assert version.parse(version_number) > version.parse('0.3.1')


def test_toi_class_construction(notebook):
    TOI = notebook.ref("TOI")
    TOI(toi_number=1)


def test_build_model(notebook):
    build_model = notebook.ref("build_model")
    build_model()


def test_posterior_saving_and_loading(notebook):
    """Save and load posteriors in HDF5"""
    test_fn = "test.h5"
    with pm.Model():
        pm.Uniform('y', 0, 20)
        trace = pm.sample(draws=10, n_init=1, chains=1, tune=10)
    save_posteriors = notebook.ref("save_posteriors")
    save_posteriors(trace, test_fn)
    assert os.path.isfile(test_fn)
    load_posterior = notebook.ref("load_posteriors")
    posterior = load_posterior(test_fn)
    assert isinstance(posterior, pd.DataFrame)
    os.remove(test_fn)
