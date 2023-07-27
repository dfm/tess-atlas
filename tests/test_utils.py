import os

import numpy as np

from tess_atlas import __website__, utils
from tess_atlas.file_management import read_last_n_lines
from tess_atlas.utils import notebook_initalisations


def test_grep_toi_number():
    toi_number = 101
    test_strings = [
        f"http://localhost:63342/tess-atlas/tests/out_webtest/html/_build/content/toi_notebooks/toi_{toi_number}.html"
        f"{__website__}/content/toi_notebooks/toi_{toi_number}.html"
        f"run_toi({toi_number})"
    ]
    for test_string in test_strings:
        assert utils.grep_toi_number(test_string) == toi_number
    assert utils.grep_toi_number("nonesense") == None


def test_notebook_initialisation(tmp_path):
    os.chdir(tmp_path)
    utils.notebook_initalisations(tmp_path)
    base_theano, compile_theano = utils.get_theano_cache(tmp_path)
    assert os.path.isdir(base_theano)
    assert str(tmp_path) in base_theano


def test_read_last_n_lines(tmp_path):
    data = np.array([i for i in range(100)])
    filepath = tmp_path / "test.txt"
    np.savetxt(filepath, data, fmt="%d")
    last_10_lines = read_last_n_lines(filepath, 10).split("\n")
    assert np.all(last_10_lines[0:9] == [str(i) for i in data[-9:]])
