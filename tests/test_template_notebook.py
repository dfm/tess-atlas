"""Module to run the entire template notebook"""

import os
import shutil
import subprocess
import unittest

import nbformat

from tess_atlas import run_toi
from tess_atlas.tess_atlas_version import __version__

SINGLE_PLANET = 103
MULTI_PLANET = 178  # has 3 planets
SINGLE_TRANSIT = 2180  # period < 0


class NotebookRunnerTestCase(unittest.TestCase):
    def setUp(self):
        self.start_dir = os.getcwd()
        self.version = "TEST"
        self.outdir = f"../notebooks/{self.version}"
        os.makedirs(self.outdir, exist_ok=True)

    def tearDown(self):
        os.chdir(self.start_dir)
        if os.path.exists(self.outdir):
            shutil.rmtree(self.outdir)

    def test_notebook_creation(self):
        notebook_fn = run_toi.create_toi_notebook_from_template_notebook(
            toi_number=SINGLE_PLANET, version=self.version, quickrun=True
        )
        self.assertTrue(os.path.exists(notebook_fn))

        try:
            with open(notebook_fn) as f:
                notebook = nbformat.read(f, as_version=4)
                nbformat.validate(notebook)
        except nbformat.ValidationError:
            self.fail(f"{notebook_fn} is an invalid notebook")

    def test_slow_notebook_execution(self):
        tmp_version = self.version
        self.version = __version__
        notebook_execution(
            SINGLE_TRANSIT, version=self.version, quickrun=False
        )
        self.version = tmp_version

    def test_quick_notebook_execution(self):
        notebook_execution(MULTI_PLANET, version=self.version, quickrun=True)


def notebook_execution(toi_id, version, quickrun=True, remove_after=False):
    notebook_fn = run_toi.create_toi_notebook_from_template_notebook(
        toi_number=toi_id, version=version, quickrun=quickrun
    )
    success = run_toi.execute_toi_notebook(notebook_fn, version=version)

    subprocess.check_call(f"git rm --cached {notebook_fn} -f", shell=True)
    samples_file = (
        f"notebooks/{version}/toi_{toi_id}_files/toi_{toi_id}.netcdf"
    )
    assert os.path.exists(samples_file), samples_file
    assert success
    if remove_after:
        shutil.rmtree(f"notebooks/{version}/toi_{toi_id}_files/")


if __name__ == "__main__":
    unittest.main()
