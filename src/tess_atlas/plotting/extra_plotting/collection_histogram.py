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
