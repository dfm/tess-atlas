import json
from typing import Dict


def save_json(fpath: str, data_dict: Dict):
    with open(fpath, "w") as f:
        json.dump(data_dict, fp=f, indent=2)


def load_json(fpath: str) -> Dict:
    with open(fpath, "r") as f:
        return json.load(f)
