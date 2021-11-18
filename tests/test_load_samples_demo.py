"""
1. create fake sample files
2. generate load_samples_demo.ipynb from `py` file
3. run load_samples_demo.ipynb notebook
4. ensure no error while running + figures created
"""
import os
import shutil

import pymc3 as pm

from tess_atlas.data import TICEntry
from tess_atlas.notebook_preprocessors import run_load_samples_demo

DATA = dict(TOI=103, TIC=336732616)
version = "TEST_LOADER_DEMO"


def get_outdir():
    outdir = os.path.join(f"./notebooks/{version}/toi_{DATA['TOI']}_files/")
    os.makedirs(outdir, exist_ok=True)
    return outdir


def create_fake_sample_files(version):
    __version__ = version
    with pm.Model():
        pm.Uniform("p[0]", 0, 20)
        pm.Uniform("b[0]", 0, 20)
        pm.Uniform("r[0]", 0, 20)
        trace = pm.sample(draws=10, n_init=1, chains=1, tune=10)
    tic_entry = TICEntry(
        tic_number=DATA["TIC"], candidates=[], toi=DATA["TOI"]
    )
    tic_entry.inference_trace = trace
    fname = os.path.join(get_outdir(), f"toi_{DATA['TOI']}.netcdf")
    tic_entry.save_inference_trace(fname)


def test_load_samples():
    create_fake_sample_files(version)
    fn = run_load_samples_demo.get_load_samples_demo_notebook_filename(version)
    assert os.path.exists(fn)
    successful_operation = run_load_samples_demo.execute_ipynb(fn)
    assert successful_operation
    shutil.rmtree(get_outdir())
