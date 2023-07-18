import logging
import os

import interruptingcow
import nbformat
from nbconvert import HTMLExporter
from ploomber_engine import execute_notebook

from ..logger import LOGGER_NAME

__all__ = ["execute_ipynb"]

DAY_IN_SEC = 60 * 60 * 24


def execute_ipynb(
    notebook_filename: str, save_html=True, timeout=DAY_IN_SEC,
    save_profiling_data=True, **kwargs
):
    """Executes a notebook and saves its executed version.

    It also caches some profiling data in the notebook metadata in the execution dir.
    And saves an HTML version of the notebook (if requested).

    :param notebook_filename: path of notebook to process
    :return: bool if notebook-preprocessing successful/unsuccessful
    """
    success = False
    runner_logger = logging.getLogger(LOGGER_NAME)
    runner_logger.info(f"Executing {notebook_filename}")
    if save_profiling_data is not False:
        profile_memory = True
        profile_runtime = True
    try:
        with interruptingcow.timeout(timeout, exception=TimeoutError):
            run_path = os.path.dirname(notebook_filename)
            execute_notebook(
                input_path=notebook_filename,
                output_path=notebook_filename,
                cwd=run_path,
                save_profiling_data=save_profiling_data,
                profile_memory=profile_memory,
                profile_runtime=profile_runtime,
                log_output=False,
                debug_later=False,
            )
            success = True
    except Exception as e:
        runner_logger.error(
            f"Preprocessing {notebook_filename} failed:\n\n {e}."
            f"Use dltr {notebook_filename.replace('.ipynb', '.dump')} to debug."
        )

    if save_html:
        __notebook_to_html(notebook_filename)

    return success


def __read_ipynb_to_nbnode(
    notebook_filename: str,
) -> nbformat.notebooknode.NotebookNode:
    with open(notebook_filename) as f:
        return nbformat.read(f, as_version=4)


def __notebook_to_html(notebook_fname):
    notebook = __read_ipynb_to_nbnode(notebook_fname)
    html_exporter = HTMLExporter(template_name="pj")
    (body, resources) = html_exporter.from_notebook_node(notebook)
    html_fname = notebook_fname.replace(".ipynb", ".html")
    with open(html_fname, "w") as f:
        f.write(body)
