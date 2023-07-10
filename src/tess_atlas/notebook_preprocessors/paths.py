from pkg_resources import resource_filename

TOI_TEMPLATE_FNAME = resource_filename(
    "tess_atlas", "notebook_templates/toi_template.py"
)
LOAD_SAMPLES_DEMO_FNAME = resource_filename(
    "tess_atlas", "notebook_templates/load_samples_demo.py"
)
MENU_PAGE_TEMPLATE_FNAME = resource_filename(
    "tess_atlas", "notebook_templates/menu_page.py"
)
TRANSIT_MODEL = resource_filename("tess_atlas", "analysis/transit_model.py")
