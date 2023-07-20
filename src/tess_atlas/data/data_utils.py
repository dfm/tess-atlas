import json
import os.path
import pickle
import time
from math import floor, log
from typing import Dict, Optional, Union

import numpy as np


def save_json(fpath: str, data_dict: Dict):
    with open(fpath, "w") as f:
        json.dump(data_dict, fp=f, indent=2)


def load_json(fpath: str) -> Dict:
    with open(fpath, "r") as f:
        return json.load(f)


def format_bytes_to_human_readable(bytes):
    lg = 0 if bytes <= 0 else floor(log(bytes, 1024))
    return f"{round(bytes / 1024 ** lg, 2)} {['B', 'KB', 'MB', 'GB', 'TB'][int(lg)]}"


def sizeof(obj, human_readable: Optional[bool] = True) -> Union[str, int]:
    """Estimates total memory usage of (possibly nested) `obj`.
    Does NOT handle circular object references!
    """
    # bytes = asizeof(obj)
    bytes = len(pickle.dumps(obj))
    if human_readable:
        return format_bytes_to_human_readable(bytes)  # str
    else:
        return bytes  # int


def residual_rms(resid):
    return np.sqrt(np.median((resid - np.median(resid)) ** 2))
