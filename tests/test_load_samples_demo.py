"""
1. create fake sample files
2. generate load_samples_demo.ipynb from `py` file
3. run load_samples_demo.ipynb notebook
4. ensure no error while running + figures created
"""

import jupytext
import pkg_resources

from tess_atlas.run_toi import execute_ipynb


def get_load_samples_demo_notebook_filename(version=None):
    """Write demo notebook in notebooks/{version}/load_samples_demo.ipynb"""
    py_filename = pkg_resources.resource_filename(
        __name__, "load_samples_demo.py"
    )
    ipynb_filename = py_filename.replace(".py", ".ipynb")
    py_pointer = jupytext.read(ipynb_filename, fmt="py:light")
    jupytext.write(py_pointer, ipynb_filename)
    return ipynb_filename


def create_fake_sample_files(version=None):
    pass


def test_load_samples():
    version = "test_version"
    create_fake_sample_files(version)
    load_samples_demo_notebook_filename = (
        get_load_samples_demo_notebook_filename()
    )
    successful_operation = execute_ipynb(
        load_samples_demo_notebook_filename, version
    )
    assert successful_operation
