"""Module to run the entire template notebook"""

import os
import re
import shutil

import pytest
import testbook
from interruptingcow import timeout
from packaging import version

from tess_atlas.notebook_controllers.controllers import TOINotebookConroller

SINGLE_PLANET = 103
MULTI_PLANET = 178  # has 3 planets
MULTI_MODAL = 273
SINGLE_TRANSIT = 1812


def test_notebook_creation(tmpdir):
    controller = TOINotebookConroller.from_toi_number(SINGLE_PLANET, tmpdir)
    controller.generate(quickrun=True)
    assert os.path.exists(controller.notebook_path)
    assert controller.valid_notebook


@pytest.mark.skip
def test_analysis_starts(tmpdir):
    controller = TOINotebookConroller.from_toi_number(SINGLE_PLANET, tmpdir)
    controller.generate(quickrun=True)

    # mock the data download stage

    print(f"Running notebook {controller.notebook_path}")
    with pytest.raises(TimeoutError):
        with timeout(5, exception=TimeoutError):
            controller.execute()
    assert controller.execution_success is False
    # the notebook only stopped because of the timeout, not some other error


@pytest.fixture(scope="module")
def toi_notebook():
    """Share kernel with the module after executing the cells with tags"""
    tags_to_execute = ["def"]
    controller = TOINotebookConroller.from_toi_number(103, "tmp")
    controller.generate(quickrun=True)

    with testbook.testbook(
        controller.notebook_path, execute=tags_to_execute
    ) as notebook:
        notebook.allow_errors = True
        notebook.execute()
        yield notebook


def test_exoplanent_import_version_number(toi_notebook):
    toi_notebook.inject(
        """
        print(xo.__version__)
        """
    )
    version_number = (
        toi_notebook.cells[-1]["outputs"][0]["text"].strip().split("\n")[0]
    )
    assert version.parse(version_number) > version.parse("0.3.1")


@pytest.mark.skip
def test_trace_saving_and_loading(toi_notebook):
    """Save and load trace from netcdf"""
    test_dir = "toi_0_files"
    toi_notebook.inject(
        """
        with pm.Model():
            pm.Uniform('y', 0, 20)
            trace = pm.sample(draws=10, n_init=1, chains=1, tune=10)
        tic_entry = TICEntry(tic=0,candidates=[],toi=0)
        tic_entry.inference_trace = trace
        tic_entry.save_inference_trace()
        tic_entry.load_inference_trace()
        print(type(tic_entry.inference_trace))
        """
    )
    assert os.path.exists("toi_0_files/toi_0.netcdf")
    stdout_cells = [
        o
        for o in toi_notebook.cells[-1]["outputs"]
        if o.get("name") == "stdout"
    ]
    out_txt = stdout_cells[0]["text"]
    assert (
        extract_substring(out_txt) == "arviz.data.inference_data.InferenceData"
    )
    shutil.rmtree(test_dir)


def extract_substring(text, pattern="'(.+?)'"):
    try:
        found = re.search(pattern, text).group(1)
    except AttributeError:
        # AAA, ZZZ not found in the original string
        found = ""  # apply your error handling
    return found
