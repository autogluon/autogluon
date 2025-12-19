import os
import sys

sys.path = [".", ".."] + sys.path

project = "AutoGluon"
release = "1.5.0"
copyright = "2025, All authors. Licensed under Apache 2.0."

author = "AutoGluon contributors"

extensions = [
    "myst_nb",  # myst-nb.readthedocs.io
    "sphinx_copybutton",  # sphinx-copybutton.readthedocs.io
    "sphinx_design",  # github.com/executablebooks/sphinx-design
    "sphinx_inline_tabs",  # sphinx-inline-tabs.readthedocs.io
    "sphinx_togglebutton",  # sphinx-togglebutton.readthedocs.io
    "sphinxext.opengraph",  # sphinxext-opengraph.readthedocs.io/en/latest/
    "sphinx.ext.autodoc",  # www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
    "sphinx.ext.autosummary",  # www.sphinx-doc.org/en/master/usage/extensions/autosummary.html
    "sphinx.ext.napoleon",  # www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
    "sphinx.ext.viewcode",  # www.sphinx-doc.org/en/master/usage/extensions/viewcode.html
    "sphinxcontrib.googleanalytics",  # github.com/sphinx-contrib/googleanalytics
]

# See https://myst-parser.readthedocs.io/en/latest/syntax/optional.html
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "substitution",
]

autosummary_generate = True
numpydoc_show_class_members = False

googleanalytics_id = "G-6XDS99SP0C"

nb_execution_mode = "force"
# nb_execution_raise_on_error=True
nb_execution_timeout = 3600
nb_merge_streams = True

nb_execution_excludepatterns = ["jupyter_execute"]

# Sphinx creates a "tags" object from the arguments specified in the "-t" option of the "sphinx-build" cmd
# This line allows AutoGluon's CI to execute a subset of our tutorial notebooks by setting the "nb_dirs_to_exec" variable
nb_dirs_to_exec = [os.path.join("tutorials", tag) for tag in tags if os.path.isdir(os.path.join("tutorials", tag))]

if len(nb_dirs_to_exec) > 0:
    nb_dirs_to_exclude = [
        dirpath
        for dirpath, _, filenames in os.walk("tutorials")
        if any(map(lambda x: x.endswith(".ipynb"), filenames)) and not dirpath.startswith(tuple(nb_dirs_to_exec))
    ]

    for nb_dir in nb_dirs_to_exclude:
        nb_execution_excludepatterns.append(os.path.join(nb_dir, "*.ipynb"))

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "_templates",
    "README.md",
    "ReleaseInstructions.md",
    "jupyter_execute",
]
master_doc = "index"
numfig = True
numfig_secnum_depth = 2
math_numfig = True
math_number_all = True

# suppress_warnings = ['misc.highlighting_failure']

html_theme = "furo"  # furo.readthedocs.io
html_theme_options = {
    "sidebar_hide_name": True,
    "light_logo": "autogluon.png",
    "dark_logo": "autogluon-w.png",
    "globaltoc_collapse": False,
}

html_sidebars = {
    "**": [
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/scroll-start.html",
        "sidebar/navigation.html",
        # 'sidebar/ethical-ads.html', # furo maintainer requests this is set if docs are hosted on readthedocs.io
        "sidebar/scroll-end.html",
        "sidebar/variant-selector.html",
    ]
}

html_favicon = "_static/favicon.ico"

html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["custom.js"]

ogp_site_url = "https://auto.gluon.ai/"
ogp_description_length = 300
ogp_site_name = "AutoGluon"
ogp_image = "https://auto.gluon.ai/dev/_static/autogluon-logo.jpg"
ogp_image_alt = "AutoGluon Logo"
ogp_type = "website"
ogp_custom_meta_tags = [
    '<meta name="twitter:card" content="summary_large_image">',
    '<meta name="twitter:site" content="@autogluon">',
    '<meta property="og:title" content="AutoGluon: Fast and Accurate ML in 3 Lines of Code">',
    '<meta property="og:description" content="With just a few lines of code, you can train and deploy high-accuracy machine learning and deep learning models on image, text, time series, and tabular data.">',
]
