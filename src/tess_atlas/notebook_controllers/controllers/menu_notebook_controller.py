from __future__ import annotations

import logging
import os

from tess_atlas.data.analysis_summary import AnalysisSummary
from tess_atlas.data.exofop import EXOFOP_DATA

from ...logger import LOGGER_NAME
from ..paths import MENU_PAGE_TEMPLATE_FNAME
from .notebook_controller import NotebookController

logger = logging.getLogger(LOGGER_NAME)


class MenuPageController(NotebookController):
    template = MENU_PAGE_TEMPLATE_FNAME

    def _get_templatized_text(self, **kwargs):
        summary_path = kwargs["summary_path"]
        summary = AnalysisSummary.from_csv(summary_path)
        n_exofop_toi = len(
            EXOFOP_DATA.get_toi_list(remove_toi_without_lk=False)
        )
        n_tess_toi = len(EXOFOP_DATA.get_toi_list(remove_toi_without_lk=True))
        txt = self._get_template_txt()
        txt = txt.replace("{{{SUMMARY_PATH}}}", f"{summary_path}")
        txt = txt.replace("{{{N_FAIL}}}", f"{summary.n_failed_analyses:,}")
        txt = txt.replace("{{{N_PASS}}}", f"{summary.n_successful_analyses:,}")
        txt = txt.replace("{{{N_TESS_ATLAS}}}", f"{n_tess_toi:,}")
        txt = txt.replace("{{{N_EXOFOP}}}", f"{n_exofop_toi:,}")
        txt = txt.replace("{{{N_NOT_STARTED}}}", f"{summary.n_not_analysed:,}")
        return txt

    def execute(self, **kwargs) -> bool:
        kwargs["save_profiling_data"] = kwargs.get(
            "save_profiling_data", False
        )
        return super().execute(**kwargs)


def run_menu_page(notebook_dir):
    summary = AnalysisSummary.load(notebook_dir)
    menu_notebook_fn = os.path.join(notebook_dir, "menu.ipynb")
    processor = MenuPageController(menu_notebook_fn)
    processor.generate(summary_path=summary.fname(notebook_dir))
    processor.execute()
    logger.info(
        f"Menu page generated [{processor.execution_success}]: {processor.notebook_path}"
    )
