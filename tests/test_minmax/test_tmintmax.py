import matplotlib.pyplot as plt
from tess_atlas.data import TICEntry
from tess_atlas.plotting import plot_raw_lightcurve

import pandas as pd

from tqdm.auto import tqdm
import os

FAILED_TOIS = "359 710 978 1192 1894 1895 2005 2151 2168 2299 2321 2326 2338 2485 2537 2670 3487 4127 4144 4344 4348 4355 4356 4358 4555 4562 4635 4706 5087 5149 5153 5235 5396 5542 5571 5581 5619 5624".split()
# FAILED_TOIS = "978".split()
FAILED_TOIS = [int(i) for i in FAILED_TOIS]

OUT = "out"
os.makedirs(OUT, exist_ok=True)

tminequal = []
has_single = []

for t in tqdm(FAILED_TOIS):
    tic = TICEntry.load(t)
    fig = plot_raw_lightcurve(tic, save=False)
    fig.savefig(f"{OUT}/lc_{tic.toi_number}.png", bbox_inches="tight")
    plt.close(fig)
