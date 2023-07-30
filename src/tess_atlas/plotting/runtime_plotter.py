from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from tess_atlas.data.exofop import EXOFOP_DATA


def plot_runtimes_histogram(df: pd.DataFrame):
    """Plot the run stats

    df: dataframe with the run stats
    Columns: ['toi', 'execution_complete', 'runtime', 'job_type', 'timestamp']
    job_type: 'setup' or 'execution'
    runtime: in seconds

    Returns:
    fig: matplotlib figure with 2 subplots:
        ax0: histogram of runtimes (in hours) coloured by success/failure ('setup' jobs only)
        ax1: histogram of runtimes (in hours) coloured by success/failure ('execution' jobs only)
    """
    fig, axs = plt.subplots(4, 1, figsize=(5, 6))
    _plot_runtime_hist(data=df, ax=axs[1], job_type="setup")
    _plot_runtime_hist(data=df, ax=axs[2], job_type="execution")
    fig.tight_layout()
    return fig, axs


def _get_unique_data_per_job_type(data, job_type: str) -> pd.DataFrame:
    """Returns a dataframe of the unique TOIs for a given job type"""
    d = data[data["job_type"] == job_type]
    d = d.sort_values(by="runtime")
    d = d.drop_duplicates(subset=["toi"], keep="last")
    return d


def _plot_runtime_hist(
    data: pd.DataFrame, ax=None, job_type: str = "execution"
):
    """Plot the runtime histogram (in hours)"""
    if ax is None:
        fig, ax = plt.subplots()
    d = _get_unique_data_per_job_type(data=data, job_type=job_type)
    d["runtime"] = d["runtime"] / 3600  # convert to hours
    bins = np.linspace(d["runtime"].min(), d["runtime"].max(), 20)
    d_true = d[d["execution_complete"] == True]["runtime"]
    d_false = d[d["execution_complete"] == False]["runtime"]
    _plot_histogram_with_collection_bin(
        ax, d_true, bins, dict(label="Pass", color="tab:green")
    )
    _plot_histogram_with_collection_bin(
        ax, d_false, bins, dict(label="Fail", color="tab:red")
    )
    _custom_legend(ax, n_pass=len(d_true), n_fail=len(d_false))
    ax.set_xlabel("Runtime (hours)")
    ax.set_ylabel("Number of TOIs")
    ax.set_title(f"Runtime histogram ({job_type} jobs only)")


def _custom_legend(ax, n_pass, n_fail):
    n_total = EXOFOP_DATA.n_tois
    n_total_run = n_pass + n_fail
    n_remain = n_total - n_total_run

    handles = [
        Rectangle((0, 0), 1, 1, color="tab:green"),
        Rectangle((0, 0), 1, 1, color="tab:red"),
        Rectangle((0, 0), 1, 1, color="grey"),
    ]
    labels = [
        f"Pass ({n_pass}, {100 * (n_pass / n_total):.2f}%)",
        f"Fail ({n_fail}, {100 * (n_fail / n_total):.2f}%)",
        f"Remaining ({n_remain}, {100 * (n_remain / n_total):.2f}%)",
    ]
    ax.legend(handles, labels, loc="upper right")


def _plot_histogram_with_collection_bin(
    ax, data: np.ndarray, bins, plt_kwargs: Dict
):
    clipped_data = np.clip(data, bins[0], bins[-1])
    ax.hist(clipped_data, bins=bins, **plt_kwargs)
    xlabels = bins[1:].astype(str)
    xlabels[-1] += "+"
    ax.set_xlim([min(bins), max(bins)])
    xticks = ax.get_xticks().tolist()
    xticks[-1] = f"+{int(xticks[-1])}"
    ax.set_xticklabels(xticks)
    return ax
