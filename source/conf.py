# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Regression Module'
copyright = '2025, Alexandre Lalle'
author = 'Alexandre Lalle'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'numpydoc'
]

autosummary_generate = True
add_module_names = False

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

html_theme = 'alabaster'
html_static_path = ['_static']

# Alabaster theme options
html_theme_options = {
    'github_user': 'Alexandre-Lalle',
    'github_repo': 'linear_regression',
    'github_banner': True,
    'github_button': True,
    'github_type': 'star',
}

