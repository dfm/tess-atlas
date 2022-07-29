from typing import Dict

import numpy as np

from .data_object import DataObject
from .lightcurve_data import LightCurveData

BJD = 2457000

CLASS_SHORTHAND = dict(
    CP="Confirmed Planet",
    EB="Eclipsing Binary",
    IS="Instrument Noise",
    KP="Known Planet",
    O="O",
    PC="Planet Candidate",
    V="Stellar Variability",
    U="Undecided",
)


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
        classification: str,
        comment: str,
        pipeline: str,
    ):
        """
        :param float toi_id: The toi number X.Y where the Y represents the
            TOI sub number (e.g. 103.1)
        :param np.ndarray time: The list of times for which we have data for
        :param float period: Planet candidate orbital period (in days)
        :param float t0: Epoch (timestamp) of the primary transit in
            TESS Barycentric Julian Date (TBJD)
        :param float depth: Planet candidate transit depth, in parts per
            million
        :param float duration: Planet candidate transit duration, in days.
        :param float snr: Planet SNR.
        :param str classification: Planet classification after group vetting
        :param str comments: Planet community comments on exofop
        """
        self.toi_id = toi_id
        self.lc = lightcurve  # lc --> lightcurve
        self.period = period
        self.t0 = t0
        self.depth = depth
        self.duration = duration
        self.snr = snr
        self.has_data_only_for_single_transit = self.__check_if_single_transit(
            period
        )
        self.classification = classification
        self.comments = comments
        self.pipeline = pipeline

        if self.has_data_only_for_single_transit:
            self.period = self.estimate_period_from_lc()

    @property
    def pipeline(self) -> str:
        return self.__pipeline

    @pipeline.setter
    def pipeline(self, pipeline):
        if "spoc".casefold() in pipeline.casefold():
            self.__pipeline = "SPOC"
        elif "qlp".casefold() in pipeline.casefold():
            self.__pipeline = "QLP"
        else:
            self.__pipeline = pipeline.casefold()

    @property
    def classification(self) -> str:
        return self.__classification

    @classification.setter
    def classification(self, classification):
        self.__classification = CLASS_SHORTHAND.get(
            classification, classification
        )

    @property
    def t0_BJD(self):
        return self.t0 + BJD

    def estimate_period_from_lc(self):
        return self.lc.time.max() - self.lc.time.min()

    def __check_if_single_transit(self, exofop_period):
        if (exofop_period <= 0.0) or np.isnan(exofop_period):
            return True
        return False

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
        if n <= 1:
            raise ValueError(
                """Number periods {} is invalid for non-single transit system:
                (np.floor(lc_tmax - tmin) / period)
                lc_tmax = {}
                tmin = {}
                spoc period = {}
                toi = {}
                """.format(
                    n, lc_tmax, self.tmin, self.period, self.toi_id
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
            classification=toi_data["TESS Disposition"],
            comments=toi_data["Comments"],
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
                    "min-max lc": f"{self.lc.time.min():.2f}-{self.lc.time.max():.2f}",
                    "Min-Max Epochs (TBJD)": f"{self.tmin:.2f} - {self.tmax:.2f}",
                    "Period estimation": self.period_estimate,
                    "Min period": self.period_min,
                    "Num Periods": self.num_periods,
                    "duration range": f"{self.duration_min:.3f} {self.duration_max:.3f}",
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

    def __repr__(self):
        d = self.to_dict(extra=True)
        d.pop("TOI")
        d = {
            k: f"{v:.2f}" if isinstance(v, float) else v for k, v in d.items()
        }
        pp = "\n".join([f"- {k}: {v}" for k, v in d.items()])
        return f"TOI {self.toi_id}: \n{pp}"
