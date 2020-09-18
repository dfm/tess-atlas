"""Module to run unittests for functions in the template notebook"""
import pytest
import testbook

TEMPLATE_NOTEBOOK = "tess_atlas/template.ipynb"


@pytest.fixture(scope='module')
def notebook():
    """Share kernel with the module after executing the cells with tags"""
    tags_to_execute = ["def"]
    with testbook.testbook(TEMPLATE_NOTEBOOK, execute=tags_to_execute) as notebook:
        notebook.allow_errors = True
        notebook.execute()
        yield notebook


def test_exoplanent_import_version_number(notebook):
    from packaging import version
    notebook.inject(
        """
        print(xo.__version__)
        """
    )
    version_number = notebook.cells[-1]['outputs'][0]['text'].strip()
    assert version.parse(version_number) > version.parse('0.3.1')


def test_toi_class_construction(notebook):
    TOI = notebook.ref("TOI")
    TOI(toi_number=1)


def test_build_model(notebook):
    build_model = notebook.ref("build_model")
    build_model()
