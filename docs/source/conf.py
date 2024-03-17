# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import opticomlib

project = 'opticomlib'
copyright = '2024, Ing. Armando P. Romeu'
author = 'Ing. Armando P. Romeu'

version = opticomlib.__version__

extensions = [
    'sphinx.ext.autodoc', 
    'sphinx.ext.coverage',
    'sphinx.ext.viewcode', 
    'sphinx.ext.napoleon',
    'matplotlib.sphinxext.plot_directive',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
]


templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'renku'
# html_theme = 'sphinx_rtd_theme'

html_static_path = ['_static']

html_logo = '_static/logo.svg'
html_favicon = '_static/favicon_laser.ico'

python_display_short_literal_types = True
autosummary_generate = True

# Napoleon settings
napoleon_numpy_docstring = True
napoleon_include_special_with_doc = True
napoleon_include_init_with_doc = True
napoleon_google_docstring = False