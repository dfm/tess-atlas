import os

import numpy as np

from tess_atlas.file_management import (
    INFERENCE_DATA_FNAME,
    LC_DATA_FNAME,
    TIC_CSV,
)
from tess_atlas.logger import LOG_FNAME
from tess_atlas.plotting.labels import THUMBNAIL_PLOT

from .toi_notebook_core import TOINotebookCore
from .toi_notebook_metadata import TOINotebookMetadata


class TOINotebookController(TOINotebookCore, TOINotebookMetadata):
    """Controller for a TOI notebook.

    This class inherits from both TOINotebookCore and TOINotebookMetadata.
    Contains methods for generating/executing the notebook and accessing/saving
    metadata associated with the notebook
    """

    def __init__(self, notebook_path: str):
        TOINotebookCore.__init__(self, notebook_path)
        TOINotebookMetadata.__init__(self, notebook_path)

    def execute(self, **kwargs) -> bool:
        """Execute the notebook."""
        status = super().execute(**kwargs)
        self.save_metadata()
        return status

    def __repr__(self):
        return f"<TOI{self.toi} NotebookController>"

    @staticmethod
    def _generate_test_notebook(
        toi_int: int, outdir: str, additional_files=True
    ) -> str:
        """Generates a test notebook for testing purposes."""
        controller = TOINotebookController.from_toi_number(toi_int, outdir)
        controller.generate(quickrun=True)
        if additional_files:
            datafiles = f"{outdir}/toi_{toi_int}_files/"
            os.makedirs(datafiles, exist_ok=True)
            for fn in [
                INFERENCE_DATA_FNAME,
                TIC_CSV,
                LC_DATA_FNAME,
                THUMBNAIL_PLOT,
            ]:
                open(f"{datafiles}/{fn}", "w").write("test")
            fake_log = "\n".join(
                " ".join(np.random.choice([*"abcdefgh "], size=100)).split()
            )
            open(f"{datafiles}/{LOG_FNAME}", "w").write(fake_log)

        return controller.notebook_path
