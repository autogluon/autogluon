
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '.')
import nbformat

project = "AutoGluon"
release = '0.6.0'
copyright = "2022, All authors. Licensed under Apache 2.0."
author = "AutoGluon contributors"

extensions = [
    # 'myst_parser',
    # 'sphinx.ext.napoleon',
    'sphinx_design',
    'sphinx_copybutton',
    # 'nbsphinx',
    'sphinx_togglebutton',
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    'sphinx.ext.autosummary',
    'myst_nb',
    #'autoapi.extension'
    ]#"sphinxcontrib.bibtex","sphinxcontrib.rsvgconverter","sphinx.ext.autodoc","sphinx.ext.viewcode"]


myst_enable_extensions = ["colon_fence", "deflist", "substitution", "html_image"]

def notebook_reads(path):
    nb = nbformat.reads(path, as_version=4)
    for cell in nb.cells:
        tags = []
        for l in cell.source.splitlines():
            if l.startswith('#@title'):
                tags.append('hide-input')
                break
        if cell.metadata.get('collapsed'):
            tags.append('hide-output')
        if len(tags) == 2:
            tags = ['hide-cell']
        if tags:
            cell.metadata['tags'] = tags
    return nb

nb_custom_formats = {
    '.ipynb': "conf.notebook_reads"
}

nb_execution_mode = "force"
nb_execution_timeout = 300
nb_execution_excludepatterns = ['jupyter_execute']
nb_merge_streams = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'README.md', 'syllabus_raw.md']
master_doc = 'index'
numfig = True
numfig_secnum_depth = 2
math_numfig = True
math_number_all = True

suppress_warnings = ['misc.highlighting_failure']

# html_title = project
# html_theme = 'sphinx_material'
html_theme = 'furo'
html_theme_options = {
    "sidebar_hide_name": True,
    "light_logo": "autogluon.png",
    "dark_logo": "autogluon-w.png",

    # 'base_url': 'http://bashtage.github.io/sphinx-material/',
    # 'repo_url': 'https://github.com/autogluon/autogluon/',
    # 'repo_name': 'AutoGluon',
    # 'google_analytics_account': 'UA-96378503-12',
    # 'html_minify': True,
    # 'css_minify': True,
    # 'nav_title': 'Practical Machine Learning',
    # 'logo_icon': '&#xe869',
    # 'globaltoc_depth': 2,
    # "color_primary": "blue",
    # 'navigation_depth': 4,
    'globaltoc_collapse': False,
    # 'master_doc': False,
    # 'nav_links': [{'href':'index', 'title':'Home', 'internal':True}],
    "light_css_variables": {
        # "admonition-font-size": "0.95rem",
        # "color-brand-content": "#3977B9",
        # "color-brand-primary": "#3977B9",
    },
    # "announcement": "Check new release 0.5 with new forecast and multi-modal modules!",

}


# html_sidebars = {
#     "**": [#"logo-text.html",
#     "globaltoc.html", "localtoc.html", "searchbox.html"]
# }

html_sidebars = {
    "**": [
        "sidebar/scroll-start.html",
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/navigation.html",
      #  "sidebar/ethical-ads.html",
        "sidebar/scroll-end.html",
        "sidebar/variant-selector.html"
    ]
}

html_static_path = ['_static']

html_favicon = '_static/favicon.png'

# html_logo = '_static/autogluon.png'

html_css_files = [
    'custom.css', 'https://at.alicdn.com/t/font_2371118_b27k2sys2hd.css'
]

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

                        self.content = ["~%s.%s" % (clazz, method) for method in methods if not method.startswith('_')]
                    if 'attributes' in self.options:
                        _, attribs = self.get_members(c, 'attribute')
                        self.content = ["~%s.%s" % (clazz, attrib) for attrib in attribs if not attrib.startswith('_')]
                finally:
                    return super(AutoAutoSummary, self).run()

        app.add_directive('autoautosummary', AutoAutoSummary)
    except BaseException as e:
        raise e
