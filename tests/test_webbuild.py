import os

import pytest

from tess_atlas.webbuilder import build_website

N_TOI_NOTEBOOKS = 5


@pytest.mark.fake_notebook_dir(n=N_TOI_NOTEBOOKS)
def test_website_generation(tmp_path, fake_notebook_dir):
    webdir = f"{tmp_path}/webdir"

    build_website(
        builddir=webdir,
        notebook_dir=fake_notebook_dir,
        rebuild=True,
        update_api_files=False,
    )
    assert os.path.exists(webdir)
    assert os.path.exists(f"{webdir}/_build/index.html")
