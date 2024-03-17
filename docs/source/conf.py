# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

import opticomlib

project = 'opticomlib'
copyright = '2024, Ing. Armando P. Romeu'
author = 'Ing. Armando P. Romeu'
release = opticomlib.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc', 
    'sphinx.ext.doctest',
    'sphinx.ext.todo', 
    'sphinx.ext.coverage',
    'sphinx.ext.viewcode', 
    'sphinx.ext.napoleon',
    'matplotlib.sphinxext.plot_directive',
    # 'numpydoc',
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

source_suffix = '.rst'
source_encoding = 'utf-8-sig'

master_doc = 'index'

exclude_patterns = ['_build']


python_display_short_literal_types = True
autosummary_generate = True

# Napoleon settings
napoleon_numpy_docstring = True
napoleon_include_special_with_doc = True
napoleon_include_init_with_doc = True
napoleon_google_docstring = False