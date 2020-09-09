"""Module to run unittests for functions in the template notebook"""
import pytest
import testbook
from tess_atlas.tess_atlas_version import __version__

TEMPLATE_NOTEBOOK = "template.ipynb"


@pytest.fixture(scope='module')
def notebook():
    """Share kernel with the module after executing the cells with tags"""
    tags_to_execute = ["imports"]
    with testbook.testbook(TEMPLATE_NOTEBOOK, execute=tags_to_execute) as notebook:
        yield notebook


def test_exoplanent_import_version_number(notebook):
    notebook.inject(
        """
        print(xo.__version__)
        """
    )
    version_number = notebook.cells[-1]['outputs'][0]['text'].strip()
    assert version_number == '0.3.2'


@pytest.mark.xfail(raises=testbook.exceptions.TestbookRuntimeError)
def test_build_model(notebook):
    notebook.execute_cell("exoplanet_models")
    build_model = notebook.ref("build_model")
    notebook.inject(
        """
        log_mass_radius_cov = [1]
        log_mass_radius_mu = 1
        """
    )

    # ideally this would not raise an error
    model = build_model(
        x=[1],
        y=[1],
        yerr=[1],
        periods=[1],
        t0s=[1],
        depths=[1],
        mask=0,
        start=0
    )
    assert model is not None
