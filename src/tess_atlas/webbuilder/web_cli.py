import argparse
from .build_pages import make_book


def get_cli_args():
    """Get the TOI number from the CLI and return it"""
    parser = argparse.ArgumentParser(prog="build webpages")
    parser.add_argument("--webdir", type=str, help="The weboutdir")
    parser.add_argument(
        "--notebooks", type=str, help="Directory with analysed notebooks"
    )
    parser.add_argument(
        "--rebuild",
        type=str,
        default="",
        help="""'hard': Rebuild from scratch (even if some webpages exist).
        'soft': Rebuild pages w/o copying notebooks again.""",
    )
    args = parser.parse_args()
    return args.webdir, args.notebooks, args.rebuild


def main():
    outdir, notebooks_dir, rebuild = get_cli_args()
    make_book(outdir=outdir, notebooks_dir=notebooks_dir, rebuild=rebuild)


if __name__ == "__main__":
    main()
