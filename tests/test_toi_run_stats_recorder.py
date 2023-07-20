import os

import numpy as np
import pytest
import scipy.stats as stats

from tess_atlas.notebook_controllers.controllers.toi_notebook_controller.toi_run_stats_recorder import (
    RUN_STATS_FILENAME,
    TOIRunStatsRecorder,
)

N = 2000


@pytest.fixture
def fake_stats_file(tmp_path) -> str:
    # runtime drawn from Poisson distribution (mean=1000, std=100, min=0, max=3000)
    runtime = stats.poisson.rvs(3600, size=N)
    for i in range(N):
        TOIRunStatsRecorder.save_stats(
            toi=np.random.uniform(100, 3000),
            success=np.random.rand() < 0.9,
            job_type="execution" if np.random.rand() < 0.5 else "setup",
            runtime=runtime[i],
            notebook_dir=str(tmp_path),
        )
    return str(tmp_path / RUN_STATS_FILENAME)


def test_plot(fake_stats_file):
    stats = TOIRunStatsRecorder(fake_stats_file)
    stats.plot()
    assert os.path.exists(fake_stats_file.replace(".csv", ".png"))
