# -*- coding: utf-8 -*-
"""Notebook to demonstrate procedure to load samples

Given a TESS-atlas version, this notebook
1. displays the samples that are stored
2. loads the stored samples
3. creates a histogram of the discovered planet parameters

"""
import matplotlib.pyplot as plt
import numpy as np


def plot():
    """Plots a scatter plot."""
    sample_x = np.random.normal(4, 0.1, 500)
    sample_y = np.random.normal(4, 0.1, 500)
    fig, ax = plt.subplots()
    ax.plot(sample_x, sample_y, ".")
    fig.show()


def main():
    plot()


if __name__ == "__main__":
    main()
