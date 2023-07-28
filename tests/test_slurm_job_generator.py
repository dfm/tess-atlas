import os
import unittest
from unittest.mock import patch

import pandas as pd

from tess_atlas.cli.make_slurm_job_cli import get_cli_args
from tess_atlas.slurm_job_generator import slurm_job_generator
from tess_atlas.slurm_job_generator.slurm_job_generator import setup_jobs

CLEAN_AFTER = False

TEST_ARRAY_SIZE = 15


class JobgenTest(unittest.TestCase):
    def setUp(self):
        self.start_dir = os.getcwd()
        self.outdir = f"test_jobgen"

        os.makedirs(self.outdir, exist_ok=True)
        self.module_loads = "module load 1"
        self.toi_list = "module load 1"
        self.toi_fn = f"{self.outdir}/toi.csv"
        self.toi_nums = [i for i in range(100, 130)]
        self.make_toi_file()
        self.toi_int = pd.read_csv(self.toi_fn).toi_numbers.values.tolist()

    def make_toi_file(self):
        df = pd.DataFrame(
            dict(
                id=[i for i, _ in enumerate(self.toi_nums)],
                toi_numbers=self.toi_nums,
            )
        )
        df.to_csv(self.toi_fn, index=False)

    def tearDown(self):
        import shutil

        if os.path.exists(self.outdir) and CLEAN_AFTER:
            shutil.rmtree(self.outdir)

    @patch(
        "tess_atlas.slurm_job_generator.slurm_job_generator.MAX_ARRAY_SIZE",
        TEST_ARRAY_SIZE,
    )
    def test_slurmfile(self):
        assert slurm_job_generator.MAX_ARRAY_SIZE == TEST_ARRAY_SIZE
        setup_jobs(self.toi_int, self.outdir, self.module_loads, False, True)

    def test_single_job_slurmfile(self):
        setup_jobs(
            toi_numbers=self.toi_nums,
            outdir=self.outdir,
            module_loads="mod 1",
            submit=False,
            clean=True,
        )

    def test_parser(self):
        get_cli_args(["--toi_number", "1"])


if __name__ == "__main__":
    unittest.main()
