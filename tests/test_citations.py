import pytest

from tess_atlas import citations


def test_packages():
    citations.print_packages()


def test_ack():
    citations.print_acknowledgements()


def test_bib():
    citations.print_bibliography()
