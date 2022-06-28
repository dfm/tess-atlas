import argparse
from .page_builder import make_book


def get_cli_args():
    """Get the TOI number from the CLI and return it"""
    parser = argparse.ArgumentParser(prog="build webpages")
    parser.add_argument("--webdir", type=str, help="The weboutdir")
    parser.add_argument(
        "--notebooks", type=str, help="Directory with analysed notebooks"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",  # False by default
        help="""flag to rebuild from scratch (even if some webpages exist).""",
    )
    parser.add_argument(
        "--add-api",
        action="store_true",  # False by default
        help="""flag to copy over files for API""",
    )
    args = parser.parse_args()

    return dict(
        builddir=args.webdir,
        notebook_dir=args.notebooks,
        rebuild=args.rebuild,
        update_api_files=args.add_api,
    )


def main():
    make_book(**get_cli_args())


if __name__ == "__main__":
    main()
