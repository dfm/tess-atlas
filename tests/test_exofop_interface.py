from typing import List

from tess_atlas.data.exofop import EXOFOP_DATA


def test_get_toi_list():
    toi_list = EXOFOP_DATA.get_toi_list()
    assert isinstance(toi_list, list)
    assert len(toi_list) > 0


def test_get_toi_numbers_for_different_categories():
    toi_numbers = EXOFOP_DATA.get_categorised_toi_lists()
    assert isinstance(toi_numbers.single_transit, List)
    assert len(toi_numbers.single_transit) > 0


def test_update():
    EXOFOP_DATA.update()