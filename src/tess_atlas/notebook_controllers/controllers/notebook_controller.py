import os
import time

import jupytext
import nbformat
import numpy as np

from ..notebook_executor import execute_ipynb


class NotebookController:
    template = None

    def __init__(self, notebook_path):
        self.notebook_path = notebook_path
        self.execution_time = np.nan  # TODO: read this from the notebook
        self.execution_success = False  # TODO: read this from the notebook
        os.makedirs(self.notebook_dir, exist_ok=True)

    def generate(self, *args, **kwargs):
        os.makedirs(os.path.dirname(self.notebook_path), exist_ok=True)
        pyfile = self.notebook_path.replace(".ipynb", ".py")
        with open(pyfile, "w") as f:
            f.write(self._get_templatized_text(**kwargs))
        self.convert_py_to_ipynb(pyfile)
        os.remove(pyfile)

    def execute(self, **kwargs) -> bool:
        if not self.notebook_exists:
            self.generate(**kwargs)
        t0 = time.time()
        self.execution_success = execute_ipynb(self.notebook_path, **kwargs)
        t1 = time.time()
        self.execution_time = t1 - t0
        return self.execution_success

    def _get_templatized_text(self, **kwargs):
        """Returns the template text with the kwargs filled in."""
        raise NotImplementedError()

    def _get_template_txt(self):
        return self.get_file_contents(self.template)

    @staticmethod
    def convert_py_to_ipynb(pyfile):
        ipynb = pyfile.replace(".py", ".ipynb")
        template_py_pointer = jupytext.read(pyfile, fmt="py:light")
        jupytext.write(template_py_pointer, ipynb)
        # ensure notebook is valid
        notebook = nbformat.read(ipynb, as_version=4)
        nbformat.validate(notebook)

    @staticmethod
    def get_file_contents(path) -> str:
        return open(path, "r").read()

    @property
    def notebook_dir(self) -> str:
        return os.path.dirname(self.notebook_path)

    @property
    def notebook_exists(self) -> bool:
        return os.path.exists(self.notebook_path)

    @property
    def valid_notebook(self) -> bool:
        try:
            with open(self.notebook_path) as f:
                notebook = nbformat.read(f, as_version=4)
                nbformat.validate(notebook)
        except nbformat.ValidationError:
            return False
        return True
