"""Module to help collect run stats for TOIs being analysed"""
import logging
import os
from datetime import datetime, timedelta

import pandas as pd

from tess_atlas.data.exofop import EXOFOP_DATA
from tess_atlas.file_management import get_file_timestamp
from tess_atlas.logger import LOGGER_NAME, timestamp
from tess_atlas.plotting.runtime_plotter import plot_runtimes_histogram

logger = logging.getLogger(LOGGER_NAME)

RUN_STATS_FILENAME = "run_stats.csv"


class TOIRunStatsRecorder:
    def __init__(self, fname: str):
        self.fname = os.path.abspath(fname)
        self.outdir = os.path.dirname(self.fname)
        if not os.path.isfile(self.fname):
            self.__init_file()
        self.file_timestamp: datetime = get_file_timestamp(
            self.fname, as_datetime=True
        )

    def __init_file(self):
        open(self.fname, "w").write(
            "toi,execution_complete,duration_in_s,job_type,timestamp\n"
        )

    def __append(self, toi: int, success: bool, job_type: str, runtime: float):
        """Append a run stat to the run stats dataframe"""
        open(self.fname, "a").write(
            f"{toi},{success},{runtime},{job_type},{timestamp()}\n"
        )

    @staticmethod
    def save_stats(
        toi: int,
        success: bool,
        job_type: str,
        runtime: float,
        notebook_dir: str,
    ):
        """Creates/Appends to a CSV the runtime and status of the TOI analysis.
        #TODO: ask ozstar admin if this is ok to do (multiple parallel jobs writing to same file)
        """
        runstats = TOIRunStatsRecorder(f"{notebook_dir}/{RUN_STATS_FILENAME}")
        file_last_modified = runstats.file_timestamp
        runstats.__append(toi, success, job_type, runtime)
        # if the filetimestamp is older than 30 minutes, then make a new plot
        # TODO: does this actually work??
        if (datetime.now() - file_last_modified) > timedelta(30 * 60):
            runstats.plot()

    @property
    def data(self) -> pd.DataFrame:
        if not hasattr(self, "_data"):
            self._data = pd.read_csv(self.fname)
            self._data["timestamp"] = pd.to_datetime(self._data["timestamp"])
            self._data = self._data.rename(
                columns={"timestamp": "end_time", "duration_in_s": "runtime"}
            )
            t1 = self._data["end_time"]
            t0 = t1 - pd.to_timedelta(self._data["runtime"], unit="s")
            self._data["start_time"] = t0
        return self._data

    def plot(self, savefig: bool = True):
        """Plot the run stats"""
        fig, axs = plot_runtimes_histogram(self.data)
        if savefig:
            fname = (
                f"{self.outdir}/{RUN_STATS_FILENAME.replace('.csv', '.png')}"
            )
            logger.info(f"Saving run stats plot to {fname}")
            fig.savefig(fname)
        return fig, axs

    def _get_counts_execution_complete_per_job_type(self) -> pd.DataFrame:
        """Returns a dataframe of the number of execution_complete per job type
                  | execution | setup
        ----------|-----------|------
        Fail      | 60        | 54
        Pass      | 446       | 44
        Remaining | 494       | 902

        """
        # only keep unique TOIs for each job type
        d = self.data.drop_duplicates(subset=["toi", "job_type"], keep="last")
        # count number of "True" and "False" 'execution_complete' per job type
        counts = d.groupby("job_type")["execution_complete"].value_counts()
        counts = counts.unstack(level=0)
        counts = counts.rename(index={True: "Pass", False: "Fail"}).to_dict()
        for k, v in counts.items():
            total = sum(v.values())
            counts[k]["Remaining"] = EXOFOP_DATA.n_tois - total
        return pd.DataFrame(counts)
