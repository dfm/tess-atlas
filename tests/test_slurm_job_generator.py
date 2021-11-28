import os
import unittest

import pandas as pd

from tess_atlas.slurm_job_generator.slurm_job_generator import (
    get_cli_args,
    setup_jobs,
)

CLEAN_AFTER = False


class JobgenTest(unittest.TestCase):
    def setUp(self):
        self.start_dir = os.getcwd()
        self.outdir = f"test_jobgen"
        os.makedirs(self.outdir, exist_ok=True)
        self.module_loads = "module load 1"
        self.toi_list = "module load 1"
        self.toi_fn = f"{self.outdir}/toi.csv"
        self.make_toi_file()

    def make_toi_file(self):
        df = pd.DataFrame(dict(id=[0, 1], toi_numbers=[100, 101]))
        df.to_csv(self.toi_fn, index=False)

    def tearDown(self):
        import shutil

        if os.path.exists(self.outdir) and CLEAN_AFTER:
            shutil.rmtree(self.outdir)

    def test_slurmfile(self):
        setup_jobs(self.toi_fn, self.outdir, self.module_loads, False)

    def test_parser(self):
        get_cli_args(["--toi_number", "1"])


if __name__ == "__main__":
    unittest.main()
