import glob
import os

import pytest

from tess_atlas.data.analysis_summary.toi_notebook_metadata import (
    INFERENCE_DATA_FNAME,
    THUMBNAIL_PLOT,
)
from tess_atlas.notebook_preprocessors.toi_notebook_generator import (
    create_toi_notebook_from_template_notebook,
)

TMP_OUTDIR = "./tmp/tess_atlas_test_notebooks"


def get_fake_notebook_path(
    toi_int, outdir=TMP_OUTDIR, additional_files=False
) -> str:
    fn = create_toi_notebook_from_template_notebook(
        toi_number=toi_int, outdir=outdir, quickrun=True, setup=False
    )
    if additional_files:
        datafiles = f"{outdir}/toi_{toi_int}_files/"
        os.makedirs(datafiles, exist_ok=True)
        open(f"{datafiles}/{INFERENCE_DATA_FNAME}", "w").write("test")
        open(f"{datafiles}/{THUMBNAIL_PLOT}", "w").write("test")
    return fn


def get_fake_notebook_dir(
    n_toi=5, outdir=TMP_OUTDIR, additional_files=True
) -> str:
    fn = ""
    for i in range(100, 100 + n_toi):
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
