# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'opticomlib'
copyright = '2024, Ing. Armando P. Romeu'
author = 'Ing. Armando P. Romeu'
release = '0.4.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc', 
    'sphinx.ext.doctest',
    'sphinx.ext.todo', 
    'sphinx.ext.coverage',
    'sphinx.ext.viewcode', 
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'sphinx_rtd_theme',
]


templates_path = ['_templates']
exclude_patterns = []

language = 'English'


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx-rtd-theme' # ['renku', 'sphinx_rtd_them']
html_static_path = ['_static']

source_suffix = '.rst'
source_encoding = 'utf-8-sig'

master_doc = 'index'

exclude_patterns = ['_build']

import os
import sys

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../../'))

python_display_short_literal_types = True
autosummary_generate = True