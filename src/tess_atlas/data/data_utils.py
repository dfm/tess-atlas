import json
from typing import Dict
import os.path
import time


def save_json(fpath: str, data_dict: Dict):
    with open(fpath, "w") as f:
        json.dump(data_dict, fp=f, indent=2)


def load_json(fpath: str) -> Dict:
    with open(fpath, "r") as f:
        return json.load(f)


def get_file_timestamp(filepath):
    if not os.path.isfile(filepath):
        return f"UNKNOWN TIME ({filepath} unrecognised file)"
    modified_time = os.path.getmtime(filepath)
    return time.ctime(modified_time)
