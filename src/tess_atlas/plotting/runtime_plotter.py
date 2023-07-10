import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def plot_histogram_with_collection_bin(ax, data, bins, plt_kwargs):
    clipped_data = np.clip(data, bins[0], bins[-1])
    ax.hist(clipped_data, bins=bins, **plt_kwargs)
    xlabels = bins[1:].astype(str)
    xlabels[-1] += "+"
    N_labels = len(xlabels)
    ax.set_xlim([min(bins), max(bins)])
    xticks = ax.get_xticks().tolist()
    xticks[-1] = f"+{int(xticks[-1])}"
    ax.set_xticklabels(xticks)
    return ax


def plot_runtimes(df, savepath=""):
    total_num_tois = len(df)
    df["time"] = df["duration"]
    fig, ax = plt.subplots()
    df_passed = df[df["STATUS"] == "PASS"]
    df_failed = df[df["STATUS"] != "PASS"]

    bins = np.linspace(0, 5, 50)
    histargs = dict(histtype="stepfilled", lw=2)
    ax = plot_histogram_with_collection_bin(
        ax,
        df_passed.time,
        bins,
        dict(
            **histargs,
            label=f"Passed ({len(df_passed)}/{len(df)})",
            edgecolor="tab:green",
            facecolor=(0.128, 0.355, 0, 0.3),
        ),
    )

    ax = plot_histogram_with_collection_bin(
        ax,
        df_failed.time,
        bins,
        dict(
            **histargs,
            label=f"Failed ({len(df_failed)}/{len(df)})",
            edgecolor="tab:red",
            facecolor=(0.255, 0.155, 0, 0.3),
        ),
    )
    legend = ax.legend()
    offset = matplotlib.text.OffsetFrom(legend, (1.0, 0.0))

    avg_time, tot_time = np.mean(df_passed.duration), np.sum(df.duration)
    text = f"Avg Time: {avg_time:.2f} Hrs\nTotal time ~{int(tot_time)} Hrs"
    ax.annotate(
        text,
        xy=(0, 0),
        size=14,
        xycoords="figure fraction",
        xytext=(0, -20),
        textcoords=offset,
        horizontalalignment="right",
        verticalalignment="top",
    )
    ax.set_xlim(left=0)
    ax.set_xlabel("Time [Hr]")

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)
