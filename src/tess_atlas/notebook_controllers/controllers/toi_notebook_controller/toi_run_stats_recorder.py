"""Module to help collect run stats for TOIs being analysed"""
import logging
import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.dates import DateFormatter
from matplotlib.patches import Rectangle

from tess_atlas.data.exofop import EXOFOP_DATA
from tess_atlas.file_management import get_file_timestamp
from tess_atlas.logger import LOGGER_NAME, timestamp

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
        """Plot the run stats
        ax0: bar of number of successful 'setup' and 'execution' jobs (where total number is total number of TOIs)
        ax1: log-binned histogram of runtimes (in hours) coloured by success/failure ('setup' jobs only)
        ax2: log-binned histogram of runtimes (in hours) coloured by success/failure ('execution' jobs only)
        ax3: start-end time of each job (coloured by success/failure)
        """
        fig, axs = plt.subplots(4, 1, figsize=(10, 10))
        self._plot_counts_execution_complete_per_job_type(ax=axs[0])
        self._plot_runtime_hist(ax=axs[1], job_type="setup")
        self._plot_runtime_hist(ax=axs[2], job_type="execution")
        self._plot_start_end_time(ax=axs[3])
        fig.tight_layout()
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

    def _plot_counts_execution_complete_per_job_type(self, ax=None):
        """Plot the number of execution_complete per job type"""
        if ax is None:
            fig, ax = plt.subplots()
        counts = self._get_counts_execution_complete_per_job_type().T
        # plot stacked bar chart, "Pass":tab:green, "Fail":tab:red, "Remaining":tab:orange
        # pass on the bottom, then fail, then remaining
        ax.bar(counts.index, counts["Pass"], color="tab:green", label="Pass")
        ax.bar(
            counts.index,
            counts["Fail"],
            bottom=counts["Pass"],
            color="tab:red",
            label="Fail",
        )
        ax.bar(
            counts.index,
            counts["Remaining"],
            bottom=counts["Pass"] + counts["Fail"],
            color="tab:orange",
            label="Remaining",
        )
        ax.set_ylabel("Number of TOIs")
        ax.set_xlabel("")
        # add one additional tick to the x-axis to make room for the legend
        ax.set_xticks(np.arange(len(counts.index) + 2))
        # add axes tucj labels
        ax.set_xticklabels(list(counts.index) + ["", ""])
        # remove axes splines but keep ticks
        sns.despine(ax=ax, left=True, bottom=True)
        ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(0.9, 1))

    def _get_unique_data_per_job_type(self, job_type: str) -> pd.DataFrame:
        """Returns a dataframe of the unique TOIs for a given job type"""
        d = self.data[self.data["job_type"] == job_type]
        d = d.sort_values(by="runtime")
        d = d.drop_duplicates(subset=["toi"], keep="last")
        return d

    def _plot_runtime_hist(self, ax=None, job_type: str = "execution"):
        """Plot the runtime histogram (in hours)"""
        if ax is None:
            fig, ax = plt.subplots()
        d = self._get_unique_data_per_job_type(job_type)
        d["runtime"] = d["runtime"] / 3600  # convert to hours
        bins = np.linspace(d["runtime"].min(), d["runtime"].max(), 20)
        d_true = d[d["execution_complete"] == True]["runtime"]
        d_false = d[d["execution_complete"] == False]["runtime"]
        ax.hist(d_true, bins=bins, alpha=0.5, color="tab:green")
        ax.hist(d_false, bins=bins, alpha=0.5, color="tab:red")
        ax.set_xlabel("Runtime (hours)")
        ax.set_ylabel("Number of TOIs")
        ax.set_title(f"Runtime histogram ({job_type} jobs only)")

    def _plot_start_end_time(self, ax=None):
        """Plot the start-end time of each job (rectangles), x-axis is time, no-yaxis (just for visualisation)"""
        if ax is None:
            fig, ax = plt.subplots()
        d = self._get_unique_data_per_job_type("execution")
        # plot rectangles for each TOI
        for i, row in d.iterrows():
            t0, t1 = row["start_time"], row["end_time"]
            ax.add_patch(
                Rectangle((t0, 0), t1 - t0, 1, color=f"C{i}", alpha=0.1)
            )
        ax.set_xlabel("Time")
        ax.set_title("Timeline of execution jobs")
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_ylim(0, 1)
        ax.set_xlim(d["start_time"].min(), d["end_time"].max())
        # use datetime for x-axis
        ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
