import argparse
import pandas as pd
from ..data.exofop import get_toi_list

from typing import Optional, List


def get_toi_numbers(toi_csv: str):
    df = pd.read_csv(toi_csv)
    return list(df.toi_numbers.values)


def make_toi_csv(fname: str, toi_numbers: Optional[List[int]] = []):
    if len(toi_numbers) == 0:
        toi_numbers = get_toi_list()
    toi_numbers = list(set([int(i) for i in toi_numbers]))
    data = pd.DataFrame(dict(toi_numbers=toi_numbers))
    data.to_csv(fname)


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
        "--outdir", help="outdir for jobs", default="notebooks"
    )
    parser.add_argument(
        "--module_loads",
        help="String containing all module loads in one line (each module separated by a space)",
    )
    parser.add_argument(
        "--submit",
        action="store_true",  # False by default
        help="Submit once files created",
    )
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

    if args.toi_csv and args.toi_number is None:
        toi_numbers = get_toi_numbers(args.toi_csv)
    elif args.toi_csv is None and args.toi_number:
        toi_numbers = [args.toi_number]
    else:
        raise ValueError(
            f"You have provided TOI CSC: {args.toi_csv}, TOI NUMBER: {args.toi_number}."
            f"You need to provide one of the two (not both)."
        )

    return toi_numbers, args.outdir, args.module_loads, args.submit, args.clean
