import logging
import shutil
from typing import List

import numpy as np
import pytest

from tess_atlas.data.exofop import EXOFOP_DATA, constants, exofop_database
from tess_atlas.data.exofop.exofop_database import ExofopDatabase
from tess_atlas.logger import LOGGER_NAME


def test_get_toi_list():
    toi_list = EXOFOP_DATA.get_toi_list()
    assert isinstance(toi_list, list)
    assert len(toi_list) > 0


def test_get_toi_numbers_for_different_categories():
    toi_numbers = EXOFOP_DATA.get_categorised_toi_lists()
    assert isinstance(toi_numbers.single_transit, List)
    assert len(toi_numbers.single_transit) > 0


@pytest.fixture
def mock_lighcurve_check(monkeypatch):
    """Mock tess_atlas.data.exofop.exofp_database._lightcurve_available(tic: int)"""

    def mock_lighcurve_check(tic: int):
        return np.nan

    monkeypatch.setattr(
        "tess_atlas.data.exofop.exofop_database._lightcurve_available",
        mock_lighcurve_check,
    )


def test_update(tmp_path, mock_lighcurve_check, monkeypatch):
    logging.getLogger(LOGGER_NAME).setLevel(logging.DEBUG)

    # mock the TIC_DATASOURCE --> TIC_CACHE
    N_LINES = 10
    tmp_csv = str(tmp_path / "cached_tic_database.csv")
    with open(exofop_database.TIC_CACHE) as old, open(tmp_csv, "w") as new:
        lines = old.readlines()
        new.writelines(lines[: 1 + N_LINES])  # write header + N_LINES

    monkeypatch.setattr(exofop_database, "TIC_DATASOURCE", tmp_csv)
    monkeypatch.setattr(exofop_database, "TIC_CACHE", tmp_csv)

    fname = str(tmp_path / "test.csv")
    exo_db = ExofopDatabase(fname=tmp_csv)
    exo_db.update(save_name=fname)
