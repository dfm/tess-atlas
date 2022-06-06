project = 'TESS Atlas"'
copyright = "2022, TESS Atlas community"
author = "the TESS Atlas community"

html_title = "TESS Atlas"
html_logo = "_static/atlas_logo.png"
html_favicon = "_static/atlas_logo.png"
release = "1"
extensions = [
    "sphinx_togglebutton",
    "sphinx_copybutton",
    "myst_nb",
    "jupyter_book",
    "sphinx_comments",
    "sphinx_external_toc",
    "sphinx.ext.intersphinx",
    "sphinx_design",
    "sphinx_book_theme",
]
# nbsphinx_execute = 'never'
jupyter_execute_notebooks = "off"
exclude_patterns = []
html_theme = "sphinx_book_theme"
html_theme_options = dict(
    repository_url="https://github.com/dfm/tess-atlas",
    use_repository_button=True,
    use_fullscreen_button=True,
    use_download_button=True,
    search_bar_text="Search the Atlas...",
)
html_static_path = ["_static"]
language = None
pygments_style = "sphinx"
html_add_permalinks = "¶"
html_sourcelink_suffix = ""
numfig = True
panels_add_bootstrap_css = False
suppress_warnings = ["myst.domains"]
