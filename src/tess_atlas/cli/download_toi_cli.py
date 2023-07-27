import argparse

from tess_atlas.api.download_analysed_toi import download_toi

PROG = "download_toi"


def __get_cli_args():
    """Get the TOI number from the CLI and return it"""
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Download analysed TOI files (notebooks, posteriors, etc.)",
        usage=f"{PROG} <toi_number>",
    )
    parser.add_argument(
        "toi_number",
        type=int,
        help="The TOI number to download data for (e.g. 103)",
    )
    args = parser.parse_args()
    return args.toi_number


def main():
    toi = __get_cli_args()
    download_toi(toi)
