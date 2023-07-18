"""This module contains the code for generating the TOI notebooks from the template notebook.
It replaces some bits of code specific to the TOI notebooks, such as the TOI number.
"""
from __future__ import annotations

import os
import re
import time
from typing import Optional, Tuple

from ....data import TICEntry
from ....file_management import PROFILING_CSV, TOI_DIR
from ....tess_atlas_version import __version__
from ....utils import grep_toi_number
from ...paths import TOI_TEMPLATE_FNAME, TRANSIT_MODEL
from ..notebook_controller import NotebookController


class TOINotebookCore(NotebookController):
    """Core functionality for a TOI notebook generation/execution."""

    template = TOI_TEMPLATE_FNAME

    def __init__(self, notebook_filename: str):
        super().__init__(notebook_filename)
        self.toi_number: int = grep_toi_number(
            os.path.basename(notebook_filename)
        )
        if self.toi_number is None:
            raise ValueError(
                f"TOI number not found in notebook filename: {notebook_filename}"
            )
        self.toi_dir = (
            f"{self.notebook_dir}/{TOI_DIR.format(toi=self.toi_number)}"
        )

    def execute(self, **kwargs) -> bool:
        kwargs["save_profiling_data"] = kwargs.get(
            "save_profiling_data", f"{self.toi_dir}/{PROFILING_CSV}"
        )
        return super().execute(**kwargs)

    @classmethod
    def from_toi_number(cls, toi_number: int, notebook_dir: str = "notebooks"):
        """Generate a TOI notebook from a template notebook and save it the notebook_dir."""
        return cls(os.path.join(notebook_dir, f"toi_{toi_number}.ipynb"))

    def generate(self, *args, **kwargs):
        """
        Generate a TOI notebook from a template notebook and save it the notebook_dir.
        Parameters
        ----------
        setup: bool, optional
            Whether to download the TOI data. Defaults to False.
        quickrun: bool, optional
            Whether to make add some 'hacks' to make the notebook run faster. Defaults to False.
        """
        super().generate(*args, **kwargs)
        if kwargs.get("setup", False) is True:
            self.__download_data()

    def _get_templatized_text(self, **kwargs):
        quickrun = kwargs.get("quickrun", False)
        txt = self._get_template_txt()
        txt = txt.replace("{{{TOINUMBER}}}", f"{self.toi_number}")
        txt = txt.replace("{{{VERSIONNUMBER}}}", f"'{__version__}'")
        txt = txt.replace(
            "{{{TRANSIT_MODEL_CODE}}}", self.get_file_contents(TRANSIT_MODEL)
        )
        txt = _quickrun_replacements(txt) if quickrun else txt
        return txt

    def __download_data(self):
        """Downloads data needed for the TOI notebook."""
        curr_dir = os.getcwd()
        os.chdir(self.notebook_dir)
        tic_data = TICEntry.load(self.toi_number)
        tic_data.save_data()
        os.chdir(curr_dir)

    @staticmethod
    def run_toi(
        toi_number: int,
        outdir: str,
        quickrun: Optional[bool] = False,
        setup: Optional[bool] = False,
    ) -> Tuple[bool, float]:
        """Creates+preprocesses TOI notebook and records the executions' stats.

        Args:
            toi_number: int
                The TOI Id number
            quickrun: bool
                If True changes sampler settings to run the notebooks faster
                (useful for testing/debugging -- produces non-scientific results)
            outdir: str
                Base outdir for TOI. Notebook will be saved at
                {outdir}/{tess_atlas_version}/toi_{toi_number}.ipynb}
            setup: bool
                If true creates notebook + downloads data needed for analysis
                but does not execute notebook

        Returns:
            execution_successful: bool
                True if successful run of notebook
            run_duration: float
                Time of analysis (in seconds)
        """
        toi_nb_processor = TOINotebookCore.from_toi_number(toi_number, outdir)
        if setup:
            t0 = time.time()
            toi_nb_processor.generate(quickrun=quickrun, setup=setup)
            execution_successful = toi_nb_processor.notebook_exists
            runtime = time.time() - t0
        else:
            execution_successful = toi_nb_processor.execute()
            runtime = toi_nb_processor.execution_time
            _record_run_stats(
                toi_number, execution_successful, runtime, outdir
            )
        return execution_successful, runtime


def _record_run_stats(
    toi_number: int,
    execution_successful: bool,
    run_duration: float,
    outdir: str,
):
    """Creates/Appends to a CSV the runtime and status of the TOI analysis.
    #TODO: ask ozstar admin if this is ok to do (multiple parallel jobs writing to same file)
    """
    fname = os.path.join(outdir, "run_stats.csv")
    if not os.path.isfile(fname):
        open(fname, "w").write("toi,execution_complete,duration_in_s\n")
    open(fname, "a").write(
        f"{toi_number},{execution_successful},{run_duration}\n"
    )


def _quickrun_replacements(txt) -> str:
    """Hacks to make the notebook run faster for testing purposes."""
    txt = txt.replace("TESS Atlas fit", "TESS Atlas fit (quickrun)")
    txt = re.sub(r"tune=[0-9]+", f"tune={5}", txt)
    txt = re.sub(r"draws=[0-9]+", f"draws={10}", txt)
    txt = re.sub(r"chains=[0-9]+", f"chains={1}", txt)
    txt = re.sub(r"cores=[0-9]+", f"cores={1}", txt)
    txt = txt.replace(
        "init_params(planet_transit_model, **params)",
        "init_params(planet_transit_model, **params, quick=True)",
    )
    txt = txt.replace("kernel = terms", "# kernel = terms")
    txt = txt.replace("gp = GaussianProcess(", "# gp = GaussianProcess(")
    txt = txt.replace("gp.marginal(name=", "# gp.marginal(name=")
    txt = txt.replace("gp.predict(residual, return_var=False)", "residual")
    txt = txt.replace(
        "# cache params",
        'obs = pm.Normal("obs", mu=residual, sigma=1, observed=residual)',
    )
    txt = txt.replace("from celerite2.theano", "# from celerite2.theano")
    return txt
