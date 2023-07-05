from tess_atlas import citations


def test_packages(capsys):
    citations.print_packages()
    packages = capsys.readouterr().out
    assert "tess-atlas" in packages
    assert "lightkurve" in packages
    assert "exoplanet" in packages


def test_ack(capsys):
    citations.print_acknowledgements()
    citations_txt = capsys.readouterr().out
    assert "exoplanet:joss" in citations_txt


def test_bib(capsys):
    citations.print_bibliography()
    bib_txt = capsys.readouterr().out
    assert "exoplanet:joss" in bib_txt
