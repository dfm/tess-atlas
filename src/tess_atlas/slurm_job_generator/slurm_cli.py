import argparse
import os
from typing import List, Optional

import pandas as pd

from ..data.exofop import get_toi_list


def get_toi_numbers(toi_csv: str):
    df = pd.read_csv(toi_csv)
    return list(df.toi_numbers.values)


def make_toi_csv(fname: str, toi_numbers: Optional[List[int]] = []):
    if len(toi_numbers) == 0:
        toi_numbers = get_toi_list()
    toi_numbers = list(set([int(i) for i in toi_numbers]))
    data = pd.DataFrame(dict(toi_numbers=toi_numbers))
    data.to_csv(fname, index=False)


def add_slurm_cli_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--module_loads",
        default="git/2.18.0 gcc/9.2.0 openmpi/4.0.2 python/3.8.5",
        help="String containing all module loads in one line (each module separated by a space)",
    )
    parser.add_argument(
        "--submit",
        action="store_true",  # False by default
        help="Submit once files created",
    )
    return parser


def get_cli_args(cli_data):
    parser = argparse.ArgumentParser(
        description="Create slurm job for analysing TOIs"
    )
    parser.add_argument(
        "--toi_csv",
        default=None,
        help="CSV with the toi numbers to analyse (csv needs a column with `toi_numbers`)",
    )
    parser.add_argument(
        "--outdir",
        help="outdir for jobs. If outdir already has analysed TOIs, than (and the kwarg 'clean' not passed), than slurm files for only the unanalysed TOIs generated)",
        default="tess_atlas_catalog",
    )
    parser = add_slurm_cli_args(parser)
    parser.add_argument(
        "--clean",
        action="store_true",  # False by default
        help="Reanalyse TOIs that have already completed analysis",
    )
    parser.add_argument(
        "--toi_number",
        type=int,
        help="The TOI number to be analysed (e.g. 103). Cannot be passed with toi-csv",
        default=None,
    )
    args = parser.parse_args(cli_data)
    os.makedirs(args.outdir, exist_ok=True)
    if args.toi_csv and args.toi_number is None:  # get TOI numbers from CSV
        toi_numbers = get_toi_numbers(args.toi_csv)
    elif args.toi_csv is None and args.toi_number:  # get single TOI number
        toi_numbers = [args.toi_number]
    else:
        toi_fname = os.path.join(args.outdir, "tois.csv")  # get all TOIs
        make_toi_csv(toi_fname)
        toi_numbers = get_toi_numbers(toi_fname)

    return toi_numbers, args.outdir, args.module_loads, args.submit, args.clean
