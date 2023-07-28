import os

import pytest

from tess_atlas.data.analysis_summary import AnalysisSummary

N_TOI_NOTEBOOKS = 5


@pytest.mark.fake_notebook_dir(n=N_TOI_NOTEBOOKS)
def test_summary(fake_notebook_dir, tmpdir, monkeypatch):
    """
    Test that the summary can be loaded from a directory of notebooks

    The dir of notebooks represents the 'n_analysed' notebooks
    (the notebook exists -- so the analysis _should_ have started)
    """

    n_successful_analyses = N_TOI_NOTEBOOKS
    n_started_analyses = N_TOI_NOTEBOOKS

    # pretend that there are 10 TOIs in EXOFOP
    monkeypatch.setattr(
        "tess_atlas.data.exofop.EXOFOP_DATA.get_toi_list",
        lambda remove_toi_without_lk=False: list(range(101, 110)),
    )

    summary = AnalysisSummary.load(notebook_dir=fake_notebook_dir, n_threads=1)
    assert summary.n_total > 0
    assert summary.n_successful_analyses == n_successful_analyses
    assert summary.n_analysed == n_started_analyses
    summary.save(tmpdir)
    assert os.path.exists(summary.fname(tmpdir))
    loaded_summary = AnalysisSummary.load(tmpdir)
    assert loaded_summary.n_total == summary.n_total
