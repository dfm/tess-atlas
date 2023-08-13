import glob
import os
from pathlib import Path

import pytest

from tess_atlas.data.exofop import EXOFOP_DATA
from tess_atlas.notebook_controllers.controllers import TOINotebookController

TMP_OUTDIR = "./tmp/tess_atlas_test_notebooks"


def get_fake_notebook_path(
    toi_int, outdir=TMP_OUTDIR, additional_files=False
) -> str:
    return TOINotebookController._generate_test_notebook(
        toi_int, outdir, additional_files
    )


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
