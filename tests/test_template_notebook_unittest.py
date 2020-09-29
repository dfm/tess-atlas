"""Module to run unittests for functions in the template notebook"""
import os
import re

import pytest
import testbook
from packaging import version

TEMPLATE_NOTEBOOK = "tess_atlas/template.ipynb"


def extract_substring(text, pattern="'(.+?)'"):
    try:
        found = re.search(pattern, text).group(1)
    except AttributeError:
        # AAA, ZZZ not found in the original string
        found = ''  # apply your error handling
    return found


@pytest.fixture(scope='module')
def notebook():
    """Share kernel with the module after executing the cells with tags"""
    tags_to_execute = ["def"]
    with testbook.testbook(TEMPLATE_NOTEBOOK, execute=tags_to_execute) as notebook:
        notebook.allow_errors = True
        notebook.execute()
        yield notebook


def test_exoplanent_import_version_number(notebook):
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


def test_trace_saving_and_loading(notebook):
    """Save and load trace from netcdf"""
    test_fn = "test.netcdf"
    notebook.inject(
        """
        with pm.Model():
            pm.Uniform('y', 0, 20)
            trace = pm.sample(draws=10, n_init=1, chains=1, tune=10)
        save_trace(trace, 'test.netcdf')
        """
    )
    assert os.path.exists("test.netcdf")
    notebook.inject(
        """
        trace = load_trace('test.netcdf')
        print(type(trace))
        """
    )
    print( notebook.cells[-1]['outputs'])
    out_txt = notebook.cells[-1]['outputs'][0]['text']
    assert extract_substring(out_txt) == 'arviz.data.inference_data.InferenceData'
    os.remove(test_fn)
