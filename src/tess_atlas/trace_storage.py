# -*- coding: utf-8 -*-
"""Stores and loads pymc3 traces to netcdf files"""
from pathlib import Path

import arviz as az
from pymc3.sampling import MultiTrace


def validate_trace_filename(filename: str):
    suffix = Path(filename).suffix
    if suffix != ".netcdf":
        raise ValueError(f"{suffix} is an invalid extension.")


def save_trace(trace: MultiTrace, filename: str):
    """Save pymc3 trace as a netcdf file"""
    validate_trace_filename(filename)
    az_trace = az.from_pymc3(trace)
    az_trace.to_netcdf(filename)


def load_trace(filename: str):
    """Load pymc3 trace from netcdf file and return an arviz InferenceData object"""
    validate_trace_filename(filename)
    return az.from_netcdf(filename)
