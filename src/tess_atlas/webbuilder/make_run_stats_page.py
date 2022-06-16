"""
TODO:
1. add to webpage
2. 'all' is not inclusive of runs that failed before logging Time
3. automate process during the building of webpages
"""

import pandas as pd
import matplotlib.pyplot as plt


def load_runtime_logs():
    pass


def plot_runtimes(fname):
    all = pd.read_csv(fname)
    all["hours"] = all["duration_in_s"] / 3600
    s_runs = runtime_data[all["execution_complete"] == True]
    f_runs = runtime_data[all["execution_complete"] == False]
    fig, ax = plt.subplots(1, 1)
    kwgs = dict(bins=50, histtype="step", lw=3)
    ax.hist(
        runtime_data["hours"],
        color="tab:blue",
        **kwgs,
        label=f"All ({len(all)})",
    )
    ax.hist(
        s_runs["hours"],
        color="tab:green",
        **kwgs,
        label=f"Successful ({len(s_runs)})",
    )
    ax.hist(
        f_runs["hours"],
        color="tab:red",
        **kwgs,
        label=f"Failed ({len(f_runs)})",
    )
    ax.set_ylabel("# TOIs")
    ax.set_xlabel("Hours")
    ax.legend()
    fig.savefig("plot.png")


plot_runtimes("run_stats.csv")
