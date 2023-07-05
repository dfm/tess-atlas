import logging
import os

import nbformat
from nbconvert import HTMLExporter
from nbconvert.exporters import export
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor
from ploomber_engine import (  # TODO: use this instead of the custom executor
    execute_notebook,
)

from tess_atlas.utils import RUNNER_LOGGER_NAME


def read_notebook(
    notebook_filename: str,
) -> nbformat.notebooknode.NotebookNode:
    """
    :param notebook_filename: path of notebook to process
    :return: notebook object
    """
    with open(notebook_filename) as f:
        notebook = nbformat.read(f, as_version=4)
    return notebook


def execute_ipynb(notebook_filename: str, save_html=True):
    """
    :param notebook_filename: path of notebook to process
    :return: bool if notebook-preprocessing successful/unsuccessful
    """
    success = True
    notebook = read_notebook(notebook_filename)

    runner_logger = logging.getLogger(RUNNER_LOGGER_NAME)
    runner_logger.info(f"Executing {notebook_filename}")

    try:
        # Note that path specifies in which folder to execute the notebook.
        run_path = os.path.dirname(notebook_filename)
        execute_notebook(
            notebook_filename,
            notebook_filename,
            cwd=run_path,
            save_profiling_data=True,
            profile_memory=True,
            profile_runtime=True,
            log_output=False,
            debug_later=True,
        )
    except Exception as e:
        runner_logger.error(
            f"Preprocessing {notebook_filename} failed:\n\n {e}."
            f"Use dltr {notebook_filename.replace('.ipynb', '.dump')} to debug."
        )

    if save_html:
        notebook_to_html(notebook_filename)

    return success


def notebook_to_html(notebook_fname):
    notebook = read_notebook(notebook_fname)
    html_exporter = HTMLExporter(template_name="pj")
    (body, resources) = html_exporter.from_notebook_node(notebook)
    html_fname = notebook_fname.replace(".ipynb", ".html")
    with open(html_fname, "w") as f:
        f.write(body)
