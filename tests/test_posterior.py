"""Module to test saving/loading of posteriors saved as HDF5 file"""
import pytest
import testbook
import pandas as pd

TEMPLATE_NOTEBOOK = "tess_atlas/template.ipynb"


@pytest.fixture(scope='module')
def notebook():
    """Share kernel with the module after executing the cells with tags"""
    tags_to_execute = ["def"]
    with testbook.testbook(TEMPLATE_NOTEBOOK, execute=tags_to_execute) as notebook:
        notebook.allow_errors = True
        notebook.execute()
        yield notebook




