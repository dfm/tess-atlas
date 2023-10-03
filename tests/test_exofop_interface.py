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
