import os
import sys

sys.path = ['.', '..'] + sys.path

project = 'AutoGluon'
release = '0.7.0'
copyright = '2023, All authors. Licensed under Apache 2.0.'
author = 'AutoGluon contributors'

extensions = [
    'myst_nb',                # myst-nb.readthedocs.io
    'sphinx_copybutton',      # sphinx-copybutton.readthedocs.io
    'sphinx_design',          # github.com/executablebooks/sphinx-design
    'sphinx_inline_tabs',     # sphinx-inline-tabs.readthedocs.io
    'sphinx_togglebutton',    # sphinx-togglebutton.readthedocs.io
    'sphinx.ext.autodoc',     # www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
    'sphinx.ext.autosummary', # www.sphinx-doc.org/en/master/usage/extensions/autosummary.html
    'sphinx.ext.napoleon',    # www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
    'sphinx.ext.viewcode',    # www.sphinx-doc.org/en/master/usage/extensions/viewcode.html
    ]

# See https://myst-parser.readthedocs.io/en/latest/syntax/optional.html
myst_enable_extensions = ['colon_fence', 'deflist', 'dollarmath', 'html_image', 'substitution']

autosummary_generate = True
numpydoc_show_class_members = False

nb_execution_mode = 'force'
# nb_execution_raise_on_error=True
nb_execution_timeout = 3600
nb_merge_streams = True

nb_execution_excludepatterns = ['jupyter_execute']

nb_dirs_to_exec = [os.path.join('tutorials', tag) for tag in tags if os.path.isdir(os.path.join('tutorials', tag))]

if len(nb_dirs_to_exec) > 0:
    nb_dirs_to_exclude = [dirpath for dirpath, _, filenames in os.walk('tutorials')
                            if any(map(lambda x: x.endswith('.ipynb'), filenames))
                               and not dirpath.startswith(tuple(nb_dirs_to_exec))]

    for nb_dir in nb_dirs_to_exclude:
        nb_execution_excludepatterns.append(os.path.join(nb_dir, '*.ipynb'))

templates_path = ['_templates']
exclude_patterns = ['_build', '_templates', 'olds', 'README.md', 'ReleaseInstructions.md', 'jupyter_execute']
master_doc = 'index'
numfig = True
numfig_secnum_depth = 2
math_numfig = True
math_number_all = True

# suppress_warnings = ['misc.highlighting_failure']

html_theme = 'furo' # furo.readthedocs.io
html_theme_options = {
    'sidebar_hide_name': True,
    'light_logo': 'autogluon.png',
    'dark_logo': 'autogluon-w.png',
    'globaltoc_collapse': False,
    # 'google_analytics_account': 'UA-XXXXX', # set to enable google analytics
}

html_sidebars = {
    '**': [
        'sidebar/brand.html',
        'sidebar/search.html',
        'sidebar/scroll-start.html',
        'sidebar/navigation.html',
        # 'sidebar/ethical-ads.html', # furo maintainer requests this is set if docs are hosted on readthedocs.io
        'sidebar/scroll-end.html',
        'sidebar/variant-selector.html'
    ]
}

html_favicon = '_static/favicon.ico'

html_static_path = ['_static']
html_css_files = ['custom.css']
html_js_files = ['custom.js']

