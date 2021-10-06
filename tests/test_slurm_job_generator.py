import os
import unittest

from tess_atlas.batch_job_generator.slurm_job_generator import make_slurm_file


class JobgenTest(unittest.TestCase):
    def setUp(self):
        self.start_dir = os.getcwd()
        self.outdir = f"test_jobgen"
        os.makedirs(self.outdir, exist_ok=True)

    def tearDown(self):
        import shutil

        if os.path.exists(self.outdir):
            shutil.rmtree(self.outdir)

    def test_slurmfile(self):
        make_slurm_file(self.outdir, [100, 101, 102], "module load 1")


if __name__ == "__main__":
    unittest.main()
