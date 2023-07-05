import os

from conftest import get_fake_notebook_path

from tess_atlas.data.analysis_summary.toi_notebook_metadata import (
    ToiNotebookMetadata,
)


def test_toi_metadata(tmp_path):
    fn = get_fake_notebook_path(101, outdir=tmp_path)
    metadata = ToiNotebookMetadata(path=fn)
    assert metadata.toi == 101
    assert metadata.analyis_completed == False

    fn = get_fake_notebook_path(101, outdir=tmp_path, additional_files=True)
    metadata = ToiNotebookMetadata(fn)
    assert metadata.toi == 101
    assert metadata.analyis_completed == True
    assert metadata.inference_object_saved == True

    fn = get_fake_notebook_path(102, outdir=tmp_path, additional_files=False)
    os.remove(fn)
    metadata = ToiNotebookMetadata(fn)
    assert metadata.toi == 102
    assert metadata.analyis_completed == False
