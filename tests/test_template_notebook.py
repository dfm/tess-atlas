"""Module to run the entire template notebook"""

import os
import shutil
import unittest

import nbformat

from tess_atlas.notebook_preprocessors import run_toi
from tess_atlas.tess_atlas_version import __version__

SINGLE_PLANET = 103
MULTI_PLANET = 178  # has 3 planets
MULTI_MODAL = 273
SINGLE_TRANSIT = 1812


class NotebookRunnerTestCase(unittest.TestCase):
    def setUp(self):
        self.start_dir = os.getcwd()
        self.outdir = f"test_notebooks"
        os.makedirs(self.outdir, exist_ok=True)

    # def tearDown(self):
    #     os.chdir(self.start_dir)
    #     if os.path.exists(self.outdir):
    #         shutil.rmtree(self.outdir)

    def test_notebook_creation(self):
        notebook_fn = run_toi.create_toi_notebook_from_template_notebook(
            toi_number=723,
            quickrun=True,
            outdir=self.outdir,
        )
        self.assertTrue(os.path.exists(notebook_fn))

        try:
            with open(notebook_fn) as f:
                notebook = nbformat.read(f, as_version=4)
                nbformat.validate(notebook)
        except nbformat.ValidationError:
            self.fail(f"{notebook_fn} is an invalid notebook")

    def test_quick_notebook_execution(self):
        notebook_execution(SINGLE_PLANET, outdir=self.outdir, quickrun=True)


def notebook_execution(toi_id, outdir, quickrun=True, remove_after=False):
    success, runtime = run_toi.run_toi(
        toi_number=toi_id,
        outdir=outdir,
        quickrun=quickrun,
        setup=False,
        make_webpage=True,
    )
    assert success
    if remove_after:
        shutil.rmtree(f"{outdir}/toi_{toi_id}_files/")


if __name__ == "__main__":
    unittest.main()
