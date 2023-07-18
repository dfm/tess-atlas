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
