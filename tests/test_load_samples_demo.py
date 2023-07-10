"""
1. create fake sample files
2. generate load_samples_demo.ipynb from `py` file
3. run load_samples_demo.ipynb notebook
4. ensure no error while running + figures created
"""
import os
import shutil

import pandas as pd
import pymc3 as pm
import pytest

from tess_atlas.data import TICEntry

# from tess_atlas.notebook_preprocessors import run_load_samples_demo

DATA = dict(TOI=103, TIC=336732616)
version = "TEST_LOADER_DEMO"


@pytest.fixture
def fake_samples_path():
    with pm.Model():
        pm.Uniform("p[0]", 0, 20)
        pm.Uniform("b[0]", 0, 20)
        pm.Uniform("r[0]", 0, 20)
        trace = pm.sample(
            draws=10, n_init=1, chains=1, tune=10, return_inferencedata=True
        )
    tic_entry = TICEntry(
        toi=DATA["TOI"],
        tic_data=pd.DataFrame(),
    )
    tic_entry.outdir = os.path.join(f"./notebooks/toi_{DATA['TOI']}_files/")
    tic_entry.save_data(inference_data=trace)


# @pytest.mark.skip(reason="The demo notebook is not yet ready.")
# def test_load_samples(fake_samples_path):
#     fn = run_load_samples_demo.get_load_samples_demo_notebook_filename()
#     assert os.path.exists(fn)
#     successful_operation = run_load_samples_demo.execute_ipynb(fn)
#     assert successful_operation
#     shutil.rmtree(os.path.dirname(fake_samples_path))
