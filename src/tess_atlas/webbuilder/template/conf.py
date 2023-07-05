project = "TESS Atlas"
copyright = "2022, TESS Atlas community"
author = "the TESS Atlas community"

master_doc = "index"
html_title = project
html_logo = "_static/atlas_logo.png"
html_favicon = "_static/atlas_logo.png"
release = "1"
extensions = [
    "myst_nb",
    "sphinx_comments",
    "sphinx_external_toc",
    "sphinx.ext.intersphinx",
    "sphinx_design",
    "sphinx_book_theme",
    "sphinx_remove_toctrees",
]
nb_execution_mode = "off"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "sphinx_book_theme"
html_theme_options = dict(
    repository_url="https://github.com/dfm/tess-atlas",
    use_repository_button=True,
    use_download_button=True,
    search_bar_text="Search the Atlas...",
    show_toc_level=1,
    collapse_navigation=True,
    show_prev_next=False,
    home_page_in_toc=True,
    toc_title=project,
    # navbar_end= ["paperbutton.html"],
)
# templates_path = ["_templates"]

html_sidebars = {"**": ["navbar-logo.html", "sbt-sidebar-nav.html"]}
html_static_path = ["_static"]

html_permalinks = True
html_sourcelink_suffix = ""
numfig = True
panels_add_bootstrap_css = True
suppress_warnings = ["myst.domains"]
html_copy_source = False
remove_from_toctrees = ["content/toi_notebooks/toi_*.ipynb"]

# https://stackoverflow.com/questions/55297443/including-notebook-with-nbsphinx-fails/70474616#70474616
