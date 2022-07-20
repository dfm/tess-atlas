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
    "sphinx_remove_toctrees",
    "sphinx_toolbox.collapse",
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
    show_toc_level=1,
    collapse_navigation=True,
    show_prev_next=False,
)
html_static_path = ["_static"]
language = None
pygments_style = "sphinx"
html_permalinks = True
html_sourcelink_suffix = ""
numfig = True
panels_add_bootstrap_css = False
suppress_warnings = ["myst.domains"]
html_copy_source = True
remove_from_toctrees = ["content/toi_notebooks/toi_*.ipynb"]

# https://stackoverflow.com/questions/55297443/including-notebook-with-nbsphinx-fails/70474616#70474616
