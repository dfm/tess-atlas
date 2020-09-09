# -*- coding: utf-8 -*-
"""Module one liner

This module does what....

Example usage:

"""
import numpy as np
import matplotlib.pyplot as plt


def plot():
    """Plots a scatter plot."""
    sample_x = np.random.normal(4, 0.1, 500)
    sample_y = np.random.normal(4, 0.1, 500)
    fig, ax = plt.subplots()
    ax.plot(sample_x, sample_y, '.')
    fig.show()


def main():
    plot()


if __name__ == "__main__":
    main()