import click

from tess_atlas.logger import setup_logger
from tess_atlas.notebook_controllers.controllers.toi_notebook_controller.toi_run_stats_recorder import (
    RUN_STATS_FILENAME,
    TOIRunStatsRecorder,
)

PROG = "plot_run_stats"


@click.command(
    name=PROG,
    help="Plot the run stats from a run_stats.csv file",
)
@click.argument(
    "filename",
    type=click.Path(exists=True),
    default=RUN_STATS_FILENAME,
)
def main(filename: str):
    """Plot the run stats from a run_stats.csv file

    Args:
        filename (str): The filename of the run_stats.csv file
    """
    setup_logger()
    TOIRunStatsRecorder(filename).plot()
