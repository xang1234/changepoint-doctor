# Configuration file for the Sphinx documentation builder.

project = "changepoint-doctor"
copyright = "2025, cpd-rs contributors"
author = "cpd-rs contributors"

extensions = [
    "myst_nb",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.mermaid",
]

# -- General -----------------------------------------------------------

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", ".venv"]
templates_path = ["_templates"]

# -- MyST-Parser -------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "substitution",
    "tasklist",
]

# -- MyST-NB (notebooks) -----------------------------------------------

nb_execution_mode = "off"

# -- sphinxcontrib-bibtex -----------------------------------------------

bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "unsrt"

# -- HTML output --------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "github_url": "https://github.com/xang1234/changepoint-doctor",
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "icon_links": [],
}

html_title = "changepoint-doctor"

# -- Intersphinx --------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}
