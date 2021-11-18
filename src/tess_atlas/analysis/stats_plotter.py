import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use(
    "https://gist.githubusercontent.com/avivajpeyi/4d9839b1ceb7d3651cbb469bc6b0d69b/raw/4ee4a870126653d542572372ff3eee4e89abcab0/publication.mplstyle"
)


TOTAL_NUM_TOIS = 4511


def plot_runtimes(data_csv):
    df = pd.read_csv(data_csv)
    df["time"] = df["duration_in_s"] / (60 * 60)
    fig, ax = plt.subplots()
    df_passed = df[df["execution_complete"] == True]
    df_failed = df[df["execution_complete"] == False]
    avg_time = np.mean(df_passed.time)
    approx_time = avg_time * TOTAL_NUM_TOIS
    print(f"Avg time: {avg_time:.2f} Hr")
    print(f"Approx time for all: {approx_time:.2f} Hr")
    ax.hist(
        df_passed.time,
        density=True,
        histtype="step",
        label=f"Passed ({len(df_passed)}/{len(df)})",
        color="tab:green",
    )
    ax.hist(
        df_failed.time,
        density=True,
        histtype="step",
        label=f"Failed ({len(df_failed)}/{len(df)})",
        color="tab:red",
    )
    legend = ax.legend()
    offset = matplotlib.text.OffsetFrom(legend, (1.0, 0.0))
    text = f"Total TOIs: {TOTAL_NUM_TOIS}\nCPU hrs all TOI ~{int(approx_time)}"
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
    ax.get_yaxis().set_visible(False)
    ax.set_xlabel("Time [Hr]")

    plt.tight_layout()
    fig.savefig(data_csv.replace(".csv", ".png"))


def main():
    if len(sys.argv) != 2:
        raise ValueError("Provide the path to the stats CSV file")
    else:
        plot_runtimes(sys.argv[1])


if __name__ == "__main__":
    main()
