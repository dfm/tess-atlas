import os

from ...data import AnalysisSummary
from ..paths import MENU_PAGE_TEMPLATE_FNAME
from .notebook_controller import NotebookController


class MenuPageController(NotebookController):
    template = MENU_PAGE_TEMPLATE_FNAME

    def _get_templatized_text(self, **kwargs):
        summary_path = kwargs["summary_path"]
        summary = AnalysisSummary.from_dir(summary_path)

        txt = self._get_template_txt()
        # why am i not using jinja2?
        txt = txt.replace("{{{SUMMARY_PATH}}}", f"{summary_path}")
        txt = txt.replace("{{{N_FAIL}}}", f"{summary.n_failed_analyses}")
        txt = txt.replace("{{{N_PASS}}}", f"{summary.n_successful_analyses}")
        txt = txt.replace("{{{N_TOTAL}}}", f"{summary.n_total}")
        return txt


def run_menu_page(notebook_dir):
    summary = AnalysisSummary.from_dir(notebook_dir)
    summary_path = summary.save_to_csv(
        os.path.join(notebook_dir, "summary.csv")
    )
    menu_notebook_fn = os.path.join(notebook_dir, "menu.ipynb")
    processor = MenuPageController(menu_notebook_fn)
    processor.generate(summary_path=summary_path)
    processor.execute()
