import argparse

from tess_atlas.data.analysis_summary import AnalysisSummary

PROG = "tess_atlas_summary"


def __get_cli_args():
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="""
        Gets the latest TESS-Atlas catalog summary (a CSV with all the TOIs and their analysis status).
        If catalog_dir is provided, it builds a new summary file.
        """,
        usage=f"{PROG} --catalog_dir <dir>",
    )
    parser.add_argument(
        "--catalog_dir",
        type=str,
        help="The directory with all analyses that you want to summarise "
        "(directory with the toi_*.ipynb and toi_*_files/)."
        "If not provided, the latest summary file will be downloaded from the TESS-Atlas website.",
        default=None,
    )
    parser.add_argument(
        "--outdir",
        type=str,
        help="The directory to save the analysis summary to",
        default=".",
    )
    parser.add_argument(
        "--n_threads",
        type=int,
        help="The number of threads to use when summarising all the TOIs.",
        default=1,
    )
    return parser.parse_args()


def main():
    args = __get_cli_args()
    AnalysisSummary.load(
        notebook_dir=args.catalog_dir,
        outdir=args.outdir,
        clean=True,
        n_threads=args.n_threads,
    ).save(args.outdir)
