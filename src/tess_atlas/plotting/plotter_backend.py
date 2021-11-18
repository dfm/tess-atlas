# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import List, Optional

from tess_atlas.data import TICEntry


class PlotterBackend(ABC):
    @staticmethod
    @abstractmethod
    def plot_folded_lightcurve(
        tic_entry: TICEntry, model_lightcurves: Optional[List[float]] = None
    ):
        pass

    @staticmethod
    @abstractmethod
    def plot_folded_lightcurve(
        tic_entry: TICEntry, model_lightcurves: Optional[List[float]] = None
    ):
        pass
