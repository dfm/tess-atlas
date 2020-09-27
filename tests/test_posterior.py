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


def test_saving_posteriors_in_hdf5_file(notebook):
   """Save posteriors in HDF5"""
   make_model = notebook.ref("make_model")
   run_model = notebook.ref("run_model")
   save_posteriors = notebook.ref("save_posteriors")
   model = make_model()
   model.run()
   model.save_posteriors("test.hdf5")
   assert os.path.isfile("test.hdf5")

def test_load_posteriors(notebook):
    """Load posteriors from HDF5 as a pandas DataFrame"""
    load_posteriors = notebook.ref("load_posteriors")
    posteriors = load_posteriors("test.hdf5")
    assert isinstance(posteriors, pd.DataFrame)


