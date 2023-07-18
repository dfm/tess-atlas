import os

from conftest import get_fake_notebook_path

from tess_atlas.notebook_controllers.controllers.toi_notebook_controller.toi_notebook_metadata import (
    TOINotebookMetadata,
)


def test_toi_metadata(tmp_path):
    fn = get_fake_notebook_path(101, outdir=tmp_path)
    metadata = TOINotebookMetadata(notebook_path=fn)
    assert metadata.toi == 101
    assert metadata.analysis_completed == False

    fn = get_fake_notebook_path(101, outdir=tmp_path, additional_files=True)
    metadata = TOINotebookMetadata(fn)
    assert metadata.toi == 101
    assert metadata.analysis_completed == True

    fn = get_fake_notebook_path(102, outdir=tmp_path, additional_files=False)
    os.remove(fn)
    metadata = TOINotebookMetadata(fn)
    assert metadata.toi == 102
    assert metadata.analysis_completed == False
