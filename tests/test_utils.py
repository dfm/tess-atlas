from tess_atlas import utils


def test_grep_toi_number():
    toi_number = 101
    test_strings = [
        f"http://localhost:63342/tess-atlas/tests/out_webtest/html/_build/content/toi_notebooks/toi_{toi_number}.html"
        f"http://catalog.tess-atlas.cloud.edu.au/content/toi_notebooks/toi_{toi_number}.html"
        f"run_toi({toi_number})"
    ]
    for test_string in test_strings:
        assert utils.grep_toi_number(test_string) == toi_number
    assert utils.grep_toi_number("nonesense") == None
