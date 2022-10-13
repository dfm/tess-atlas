import os
import unittest

from tess_atlas.notebook_preprocessors.toi_notebook_generator import (
    create_toi_notebook_from_template_notebook,
)
from tess_atlas.tess_atlas_version import __version__
from tess_atlas.webbuilder.page_builder import make_book


class TestWebbuild(unittest.TestCase):
    def setUp(self) -> None:
        self.notebook_dir = "out_webtest/notebooks"
        self.webdir = "out_webtest/html"
        self.generate_fake_notebooks()

    def test_makebook(self):
        make_book(
            builddir=self.webdir,
            notebook_dir=f"{self.notebook_dir}/{__version__}/",
            rebuild=True,
            update_api_files=True,
        )

    def generate_fake_notebooks(self):
        """Generate fake notebooks for testing"""
        os.makedirs(self.notebook_dir, exist_ok=True)
        for i in [101, 102, 103, 178, 273, 1812]:
            create_toi_notebook_from_template_notebook(
                toi_number=i,
                outdir=self.notebook_dir,
                quickrun=False,
                setup=False,
            )


if __name__ == "__main__":
    unittest.main()
