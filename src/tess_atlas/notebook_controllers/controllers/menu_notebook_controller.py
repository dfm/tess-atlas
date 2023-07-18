from __future__ import annotations

import logging
import os

from ...data.analysis_summary import AnalysisSummary
from ...logger import LOGGER_NAME
from ..paths import MENU_PAGE_TEMPLATE_FNAME
from .notebook_controller import NotebookController

logger = logging.getLogger(LOGGER_NAME)


class MenuPageController(NotebookController):
    template = MENU_PAGE_TEMPLATE_FNAME

    def _get_templatized_text(self, **kwargs):
        summary_path = kwargs["summary_path"]
        summary = AnalysisSummary.load_from_csv(summary_path)
        txt = self._get_template_txt()
        txt = txt.replace("{{{SUMMARY_PATH}}}", f"{summary_path}")
        txt = txt.replace("{{{N_FAIL}}}", f"{summary.n_failed_analyses}")
        txt = txt.replace("{{{N_PASS}}}", f"{summary.n_successful_analyses}")
        txt = txt.replace("{{{N_TOTAL}}}", f"{summary.n_total}")
        return txt

    def execute(self, **kwargs) -> bool:
        kwargs["save_profiling_data"] = kwargs.get(
            "save_profiling_data", False
        )
        return super().execute(**kwargs)


def run_menu_page(notebook_dir):
    summary = AnalysisSummary.from_dir(notebook_dir)
    summary_path = summary.save_to_csv(
        os.path.join(notebook_dir, "summary.csv")
    )
    menu_notebook_fn = os.path.join(notebook_dir, "menu.ipynb")
    processor = MenuPageController(menu_notebook_fn)
    processor.generate(summary_path=summary_path)
    processor.execute()
    logger.info(
        f"Menu page generated [{processor.execution_success}]: {processor.notebook_path}"
    )
