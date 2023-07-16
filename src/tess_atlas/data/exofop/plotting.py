import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from .constants import LK_AVAIL, TIC_CACHE, TOI_INT


def plot_lk_status(new: pd.DataFrame, old: pd.DataFrame = None) -> plt.Figure:
    """Plot if the TOI has lightcurve data using the old and new TIC caches"""
    if len(old) > 0:
        fig, axes = plt.subplots(
            2, 1, figsize=(10, 5), sharex=True, sharey=True
        )
        _add_lk_status_to_ax(axes[1], old, "Old Cache")
        ax = axes[0]
    else:
        fig, axes = plt.subplots(1, 1, figsize=(10, 2.5))
        ax = axes
    _add_lk_status_to_ax(ax, new, label="Cache")
    fig.suptitle("TOIs with 2-min Lightcurve data", fontsize="xx-large")
    fig.tight_layout()
    fig.savefig(TIC_CACHE.replace(".csv", ".png"))
    return fig


def _add_lk_status_to_ax(ax: plt.Axes, data: pd.DataFrame, label=""):
    r = dict(ymin=0, ymax=2, lw=0.1, alpha=0.5)
    valid = set(data[data[LK_AVAIL] == True][TOI_INT].tolist())
    invalid = set(data[data[LK_AVAIL] == False][TOI_INT].tolist())
    nans = set(data[data[LK_AVAIL].isna()][TOI_INT].tolist())
    total = len(data)
    colors = dict(valid="tab:green", invalid="tab:red", nans="tab:orange")
    ax.vlines(
        list(valid),
        **r,
        label=f"Valid ({len(valid)}/{total} TOIs)",
        color=colors["valid"],
    )
    ax.vlines(
        list(invalid),
        **r,
        label=f"Invalid ({len(invalid)}/{total} TOIs)",
        color=colors["invalid"],
        zorder=-10,
    )
    ax.vlines(
        list(nans),
        **r,
        label=f"Nans ({len(nans)}/{total} TOIs)",
        color=colors["nans"],
        zorder=-20,
    )

    # make legend with larger markers visible in the plot
    handles, labels = ax.get_legend_handles_labels()
    handles = [Line2D([0], [0], color=c, lw=2) for c in colors.values()]
    labels = [
        f"{l} ({len(s)})" for l, s in zip(labels, [valid, invalid, nans])
    ]
    ax.legend(handles, labels, loc="upper left", fontsize=8)
    ax.set_ylim(0.99, 1.01)
    ax.set_xlim(left=100, right=max(data["TOI int"]))
    ax.set_yticks([])
    ax.set_xlabel("TOI Number")
    ax.set_ylabel(label)
