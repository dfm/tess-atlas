import json
from typing import Dict, Optional, Union
import os.path
import time
from math import floor, log
import numpy as np

# from pympler.asizeof import asizeof
import pickle


def save_json(fpath: str, data_dict: Dict):
    with open(fpath, "w") as f:
        json.dump(data_dict, fp=f, indent=2)


def load_json(fpath: str) -> Dict:
    with open(fpath, "r") as f:
        return json.load(f)


def get_file_timestamp(filepath) -> str:
    if not os.path.isfile(filepath):
        return f"UNKNOWN TIME ({filepath} unrecognised file)"
    modified_time = os.path.getmtime(filepath)
    return time.ctime(modified_time)


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
