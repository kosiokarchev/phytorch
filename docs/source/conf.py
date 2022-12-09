import datetime
import os
import sys
from typing import Literal


# -- Path setup --------------------------------------------------------------

# for extensions
sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------
import phytorch


project = 'phytorch'
copyright = f'{datetime.datetime.now().year}, Kosio Karchev'
author = 'Kosio Karchev'
version = release = str(phytorch.__version__)

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.intersphinx',
    'sphinx.ext.autodoc',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx_copybutton',
    'myst_parser',
    'nbsphinx', 'sphinx_gallery.load_style',
    'IPython.sphinxext.ipython_console_highlighting',

    # 'autoapi.extension',

    '_ext.regex_lexer',
    '_ext.any_tilde'
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'astropy': ('https://docs.astropy.org/en/stable/', None),
    'torch': ('https://pytorch.org/docs/stable', None),
    'pyro': ('http://docs.pyro.ai/en/stable/', None),
}

napoleon_google_docstring = False

todo_include_todos = True

autodoc_inherit_docstrings: bool = False
autodoc_class_signature: Literal['mixed', 'separated'] = 'separated'
autodoc_typehints: Literal['signature', 'description', 'both', 'none'] = 'description'
autodoc_typehints_description_target: Literal['all', 'documented', 'documented_params'] = 'documented_params'
autodoc_member_order: Literal['alphabetical', 'groupwise', 'bysource'] = 'groupwise'
autodoc_mock_imports: list[str] = []
autodoc_type_aliases = {
    '_GQuantity': 'unitful'
    # '_TN': 'Tensor-like'
}


# AUTOAPI
# -------
# autoapi_dirs = ['../../../phytorch/phytorch']
# autoapi_file_patterns = ['*.pyi', '*.py']
# autoapi_ignore = [
#     '**/phytorch/units/_si/*'
#     # '**/clipppy/contrib.py',
#     # '**/clipppy/autocli.py',
#     # '**/clipppy/cli.py',
#     # '**/clipppy/distributions/*',
#     # '**/clipppy/_clipppy.py',
#     # '**/clipppy/**/sampler.py'
# ]
# autoapi_options = [
#     'members', 'undoc-members', 'private-members', 'special-members',
#     # 'show-inheritance',
#     # 'show-inheritance-diagram',
# ]
# autoapi_member_order = 'groupwise'
# autoapi_keep_files = True
# autoapi_template_dir = '_autoapi_templates'

nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg'}",
    "--InlineBackend.rc=figure.dpi=96"
]
nbsphinx_execute = 'auto'
nbsphinx_codecell_lexer = 'python3'
nbsphinx_allow_errors = True

trim_footnote_reference_space = True

# ----------------------------------------------------------------------------
templates_path = ['_templates']
root_doc = 'index'
exclude_patterns = []

# language=rst
rst_prolog = nbsphinx_prolog = '''
.. |phytorch| replace:: φtorch

.. |torchdiffeq| replace:: `torchdiffeq`_
.. _torchdiffeq: https://github.com/rtqichen/torchdiffeq

.. default-role:: any
.. highlight:: python3

.. |Python| replace:: Python
.. |pytorch| replace:: `PyTorch`_
.. _pytorch: https://pytorch.org/
.. |astropy| replace:: `AstroPy`_
.. _astropy: https://www.astropy.org/

.. |citation needed| replace:: `[citation needed]`:superscript:
.. role:: strike
    :class: strike
.. role:: underline
    :class: underline

.. role:: arg(literal)

.. role:: raw-html(raw)
    :format: html

.. role:: shell(code)
    :class: highlight
    :language: shell
.. role:: python(code)
    :class: highlight
    :language: python3
'''

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
html_css_files = ['style.css']
html_js_files = ['script.js']
