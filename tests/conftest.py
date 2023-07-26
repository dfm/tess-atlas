import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from tess_atlas.data.exofop import EXOFOP_DATA
from tess_atlas.file_management import (
    INFERENCE_DATA_FNAME,
    LC_DATA_FNAME,
    TIC_CSV,
)
from tess_atlas.logger import LOG_FNAME
from tess_atlas.notebook_controllers.controllers import TOINotebookController
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
        for fn in [
            INFERENCE_DATA_FNAME,
            TIC_CSV,
            LC_DATA_FNAME,
            THUMBNAIL_PLOT,
        ]:
            open(f"{datafiles}/{fn}", "w").write("test")
        fake_log = "\n".join(
            " ".join(np.random.choice([*"abcdefgh "], size=100)).split()
        )
        open(f"{datafiles}/{LOG_FNAME}", "w").write(fake_log)

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
def fake_notebook_dir(tmp_path, n_toi=5) -> str:
    notebooks = glob.glob(f"{tmp_path}/toi_*.ipynb")
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


@pytest.fixture
def mock_exofop_get_toi_list(monkeypatch):
    """mock EXOFOP_DATA.get_toi_list(remove_toi_without_lk=True)"""

    def mock_toi_list(*args, **kwargs):
        return [i for i in range(101, 150)]

    monkeypatch.setattr(EXOFOP_DATA, "get_toi_list", mock_toi_list)
