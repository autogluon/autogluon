import sys

import nbformat

sys.path = ['.', '..'] + sys.path

project = 'AutoGluon'
release = '0.6.2'
copyright = '2023, All authors. Licensed under Apache 2.0.'
author = 'AutoGluon contributors'

extensions = [
    'myst_nb',                # myst-nb.readthedocs.io
    'sphinx_copybutton',      # sphinx-copybutton.readthedocs.io
    'sphinx_design',          # github.com/executablebooks/sphinx-design
    'sphinx_togglebutton',    # sphinx-togglebutton.readthedocs.io
    'sphinx.ext.autodoc',     # www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
    'sphinx.ext.autosummary', # www.sphinx-doc.org/en/master/usage/extensions/autosummary.html
    'sphinx.ext.viewcode',    # www.sphinx-doc.org/en/master/usage/extensions/viewcode.html
    ]

# See https://myst-parser.readthedocs.io/en/latest/syntax/optional.html
myst_enable_extensions = ['colon_fence', 'deflist', 'substitution', 'html_image']

nb_execution_mode = 'force'
nb_execution_timeout = 1200
nb_execution_excludepatterns = ['jupyter_execute']
nb_merge_streams = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'README.md', 'ReleaseInstructions.md']
master_doc = 'index'
numfig = True
numfig_secnum_depth = 2
math_numfig = True
math_number_all = True

# suppress_warnings = ['misc.highlighting_failure']

html_theme = 'furo'
html_theme_options = {
    'sidebar_hide_name': True,
    'light_logo': 'autogluon.png',
    'dark_logo': 'autogluon-w.png',
    'globaltoc_collapse': False,
    # 'google_analytics_account': 'UA-XXXXX', # set to enable google analytics
}

html_sidebars = {
    '**': [
        'sidebar/scroll-start.html',
        'sidebar/brand.html',
        'sidebar/search.html',
        'sidebar/navigation.html',
        # 'sidebar/ethical-ads.html',
        'sidebar/scroll-end.html',
        'sidebar/variant-selector.html'
    ]
}

html_static_path = ['_static']

html_favicon = '_static/favicon.png'

# html_logo = '_static/autogluon.png'

html_css_files = [
    'custom.css', 'https://at.alicdn.com/t/font_2371118_b27k2sys2hd.css'
]

'''
def setup(app):
    try:
        from sphinx.ext.autosummary import Autosummary
        from sphinx.ext.autosummary import get_documenter
        from docutils.parsers.rst import directives
        from sphinx.util.inspect import safe_getattr
        import re

        class AutoAutoSummary(Autosummary):

            option_spec = {
                'methods': directives.unchanged,
                'attributes': directives.unchanged
            }

            required_arguments = 1

            @staticmethod
            def get_members(obj, typ, include_public=None):
                if not include_public:
                    include_public = []
                items = []
                for name in dir(obj):
                    try:
                        documenter = get_documenter(app, safe_getattr(obj, name), obj)
                    except AttributeError:
                        continue
                    if documenter.objtype == typ:
                        items.append(name)
                public = [x for x in items if x in include_public or not x.startswith('_')]
                return public, items

            def run(self):
                clazz = self.arguments[0]
                try:
                    (module_name, class_name) = clazz.rsplit('.', 1)
                    m = __import__(module_name, globals(), locals(), [class_name])
                    c = getattr(m, class_name)
                    if 'methods' in self.options:
                        _, methods = self.get_members(c, 'method', ['__init__'])

                        self.content = ['~%s.%s' % (clazz, method) for method in methods if not method.startswith('_')]
                    if 'attributes' in self.options:
                        _, attribs = self.get_members(c, 'attribute')
                        self.content = ['~%s.%s' % (clazz, attrib) for attrib in attribs if not attrib.startswith('_')]
                finally:
                    return super(AutoAutoSummary, self).run()

        app.add_directive('autoautosummary', AutoAutoSummary)
    except BaseException as e:
        raise e
'''
