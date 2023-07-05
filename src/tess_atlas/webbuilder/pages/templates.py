from jinja2 import Template

TOI_LINK = Template("`TOI {{toi_int}}  <toi_notebooks/{{toi_fname}}.html>`_")

IMAGE = Template(
    """.. figure:: toi_notebooks/{{rel_path}}
            :scale: 50 %
            :target: toi_notebooks/{{toi_fname}}.html

"""
)


def render_page_template(fname, page_data):
    with open(fname) as file_:
        template = Template(file_.read())
    return template.render(**page_data)
