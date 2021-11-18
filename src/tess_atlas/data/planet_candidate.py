from typing import Dict

import numpy as np

from .data_object import DataObject
from .lightcurve_data import LightCurveData


def calculate_time_fold(t, t0, p):
    """Function to get time-fold"""
    hp = 0.5 * p
    return (t - t0 + hp) % p - hp


class PlanetCandidate(DataObject):
    """Plant Candidate obtained by TESS."""

    def __init__(
        self,
        toi_id: float,
        period: float,
        time: np.ndarray,
        t0: float,
        depth: float,
        duration: float,
    ):
        """
        :param float toi_id: The toi number X.Y where the Y represents the
            TOI sub number (e.g. 103.1)
        :param np.ndarray time: The list of times for which we have data for
        :param float period: Planet candidate orbital period (in days)
        :param float t0: Epoch (timestamp) of the primary transit in
            Barycentric Julian Date
        :param float depth: Planet candidate transit depth, in parts per
            million
        :param float duration: Planet candidate transit duration, in days.
        """
        self.toi_id = toi_id
        self.__time = time
        self.has_data_only_for_single_transit = False
        self.period = period
        self.t0 = t0
        self.depth = depth
        self.duration = duration

    @property
    def period(self):
        return self.__period

    @period.setter
    def period(self, p):
        if (p <= 0.0) or np.isnan(p):
            self.has_data_only_for_single_transit = True
            self.__period = self.__time.max() - self.__time.min()
        else:
            self.__period = p

    @property
    def index(self):
        return int(str(self.toi_id).split(".")[1])

    @property
    def num_periods(self):
        """number of periods between t0 and tmax"""
        if self.has_data_only_for_single_transit:
            return 0
        return (np.floor(max(self.__time) - self.t0) / self.period).astype(int)

    @property
    def tmax(self):
        """Time of the last transit"""
        return self.t0 + self.num_periods * self.period

    @property
    def period_min(self):
        """the minimum possible period"""
        return np.maximum(
            np.abs(self.t0 - self.__time.max()),
            np.abs(self.__time.min() - self.t0),
        )

    @property
    def duration_max(self):
        if self.has_data_only_for_single_transit:
            return 1.0
        return max(1.5 * self.duration, 0.1)

    @property
    def duration_min(self):
        return min(self.duration, 2 * np.min(np.diff(self.__time)))

    @classmethod
    def from_database(cls, toi_data: Dict, lightcurve: LightCurveData):
        unpack_data = dict(
            toi_id=toi_data["TOI"],
            period=toi_data["Period (days)"],
            t0=toi_data["Epoch (BJD)"] - 2457000,  # convert to TBJD
            depth=toi_data["Depth (ppm)"]
            * 1e-3,  # convert to parts per thousand
            duration=toi_data["Duration (hours)"] / 24.0,  # convert to days,
            time=lightcurve.time,
        )
        return cls(**unpack_data)

    @classmethod
    def from_cache(cls, toi_data: Dict, lightcurve: LightCurveData):
        return cls.from_database(toi_data, lightcurve)

    def save_data(self):
        """Saving done by Tic Entry"""
        pass

    def get_timefold(self, t):
        """Used in plotting"""
        return calculate_time_fold(t, self.t0, self.period)

    def to_dict(self):
        return {
            "TOI": self.toi_id,
            "Period (days)": self.period,
            "Epoch (TBJD)": self.t0,
            "Depth (ppt)": self.depth,
            "Duration (days)": self.duration,
            "Single Transit": self.has_data_only_for_single_transit,
        }
