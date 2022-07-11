from typing import Dict

import numpy as np

from .data_object import DataObject
from .lightcurve_data import LightCurveData


class PlanetCandidate(DataObject):
    """Plant Candidate obtained by TESS."""

    def __init__(
        self,
        toi_id: float,
        period: float,
        t0: float,
        depth: float,
        duration: float,
        snr: float,
        lightcurve: LightCurveData,
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
        :param float snr: Planet SNR.
        """
        self.toi_id = toi_id
        self.lc = lightcurve  # lc --> lightcurve
        self.has_data_only_for_single_transit = False
        self.period = period
        self.t0 = t0
        self.depth = depth
        self.duration = duration
        self.snr = snr

    @property
    def period(self):
        return self.__period

    @period.setter
    def period(self, p):
        if (p <= 0.0) or np.isnan(p):
            self.has_data_only_for_single_transit = True
            self.__period = self.lc.time.max() - self.lc.time.min()
        else:
            self.__period = p

    @property
    def index(self):
        return int(str(self.toi_id).split(".")[1])

    @property
    def num_periods(self):
        """number of periods between tmin and tmax"""
        if self.has_data_only_for_single_transit:
            return 0
        lc_tmax = max(self.lc.time)
        n = np.floor(lc_tmax - self.tmin) / self.period
        if n <= 0:
            raise ValueError(
                """Number periods {} is invalid:
                (np.floor(lc_tmax - tmin) / period)
                lc_tmax = {}
                tmin = {}
                spoc period = {}
                """.format(
                    n, lc_tmax, self.tmin, self.period
                )
            )
        return n.astype(int)

    @property
    def tmax(self):
        """Time of the last transit"""
        tlast = self.num_periods * self.period
        return self.tmin + tlast

    @property
    def tmin(self):
        """Time of the first transit (t0 might be a future epoch)"""
        lc_min = min(self.lc.time)
        return (
            self.t0 + np.ceil((lc_min - self.t0) / self.period) * self.period
        )

    @property
    def period_min(self):
        """the minimum possible period"""
        lc_min = min(self.lc.time)
        lc_max = max(self.lc.time)
        return np.maximum(
            np.abs(self.t0 - lc_max),
            np.abs(lc_min - self.t0),
        )

    @property
    def period_estimate(self):
        """period estimate from tmin tmax num_periods"""
        return (self.tmax - self.tmin) / self.num_periods

    @property
    def duration_max(self):
        if self.has_data_only_for_single_transit:
            return 1.0
        return max(10 * self.duration, 0.1)

    @property
    def duration_min(self):
        return min(0.1 * self.duration, 2 * self.lc.cadence)

    @classmethod
    def from_database(cls, toi_data: Dict, lightcurve: LightCurveData):
        unpack_data = dict(
            toi_id=toi_data["TOI"],
            period=toi_data["Period (days)"],
            t0=toi_data["Epoch (BJD)"] - 2457000,  # convert to TBJD
            depth=toi_data["Depth (ppm)"]
            * 1e-3,  # convert to parts per thousand
            duration=toi_data["Duration (hours)"] / 24.0,  # convert to days,
            lightcurve=lightcurve,
            snr=toi_data["Planet SNR"],
        )
        return cls(**unpack_data)

    @classmethod
    def from_cache(cls, toi_data: Dict, lightcurve: LightCurveData):
        return cls.from_database(toi_data, lightcurve)

    def save_data(self):
        """Saving done by Tic Entry"""
        pass

    def to_dict(self, extra=False):
        data = {
            "TOI": self.toi_id,
            "Period (days)": self.period,
            "Epoch (TBJD)": self.t0,
            "Depth (ppt)": self.depth,
            "Duration (days)": self.duration,
            "Planet SNR": self.snr,
            "Single Transit": self.has_data_only_for_single_transit,
        }
        if extra:
            data.update(
                {
                    "Min-Max Epochs (TBJD)": f"{self.tmin:.2f} - {self.tmax:.2f}",
                    "Period estimation": self.period_estimate,
                    "Num Periods": self.num_periods,
                    "duration range": f"{self.duration_min} {self.duration_max}",
                }
            )
        return data

    def _repr_html_(self):
        d = self.to_dict()
        d.pop("TOI")
        d = {
            k: f"{v:.2f}" if isinstance(v, float) else v for k, v in d.items()
        }
        html = "\n".join([f"<li>{k}: {v}</li>" for k, v in d.items()])
        return f"TOI {self.toi_id}: \n" "<ul>\n" f"{html}" "</ul>"
