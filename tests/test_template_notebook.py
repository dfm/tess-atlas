"""Module to run the entire template notebook"""

TEMPLATE_NOTEBOOK = "template.ipynb"

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def test_run_entire_notebook():
    with open(TEMPLATE_NOTEBOOK) as f:
        notebook = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=-1)
    ep.preprocess(notebook)

