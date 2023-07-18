import glob
import os
from pathlib import Path

import pytest

from tess_atlas.file_management import INFERENCE_DATA_FNAME
from tess_atlas.notebook_controllers.controllers.toi_notebook_controller import (
    TOINotebookController,
)
from tess_atlas.plotting.labels import THUMBNAIL_PLOT

TMP_OUTDIR = "./tmp/tess_atlas_test_notebooks"


def get_fake_notebook_path(
    toi_int, outdir=TMP_OUTDIR, additional_files=False
) -> str:
    controller = TOINotebookController.from_toi_number(toi_int, outdir)
    controller.generate(quickrun=True)
    if additional_files:
        datafiles = f"{outdir}/toi_{toi_int}_files/"
        os.makedirs(datafiles, exist_ok=True)
        open(f"{datafiles}/{INFERENCE_DATA_FNAME}", "w").write("test")
        open(f"{datafiles}/{THUMBNAIL_PLOT}", "w").write("test")
    return controller.notebook_path


def get_fake_notebook_dir(
    n_toi=5, outdir=TMP_OUTDIR, additional_files=True
) -> str:
    fn = ""
    for i in range(101, 101 + n_toi):
        fn = get_fake_notebook_path(
            i, outdir=outdir, additional_files=additional_files
        )
    return os.path.dirname(fn)


@pytest.fixture
def fake_notebook_dir(n_toi=5) -> str:
    notebooks = glob.glob(f"{TMP_OUTDIR}/toi_*.ipynb")
    if len(notebooks) >= n_toi:
        return os.path.dirname(notebooks[0])
    else:
        return get_fake_notebook_dir(
            n_toi=n_toi, outdir=TMP_OUTDIR, additional_files=True
        )


@pytest.fixture
def tmp_working_dir(tmp_path) -> str:
    """
    Create temporary path using pytest native fixture,
    them move it, yield, and restore the original path
    """
    old = os.getcwd()
    os.chdir(str(tmp_path))
    yield str(Path(tmp_path).resolve())
    os.chdir(old)
