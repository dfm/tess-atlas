from pkg_resources import resource_filename

TOI_TEMPLATE_FNAME = resource_filename(
    "tess_atlas", "notebook_templates/toi_template.py"
)
TRANSIT_MODEL = resource_filename("tess_atlas", "analysis/transit_model.py")
