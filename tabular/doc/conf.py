"""Sphinx configuration."""
from __future__ import print_function

import datetime
import os
import shutil


def run_apidoc(app):
    """Generate doc stubs using sphinx-apidoc."""
    module_dir = os.path.join(app.srcdir, "../src/")
    output_dir = os.path.join(app.srcdir, "_apidoc")
    excludes = []

    # Ensure that any stale apidoc files are cleaned up first.
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    cmd = [
        "--separate",
        "--module-first",
        "--doc-project=API Reference",
        "-o",
        output_dir,
        module_dir,
    ]
    cmd.extend(excludes)

    try:
        from sphinx.ext import apidoc  # Sphinx >= 1.7

        apidoc.main(cmd)
    except ImportError:
        from sphinx import apidoc  # Sphinx < 1.7

        cmd.insert(0, apidoc.__file__)
        apidoc.main(cmd)


def setup(app):
    """Register our sphinx-apidoc hook."""
    app.connect("builder-inited", run_apidoc)


# Sphinx configuration below.
project = "Tabular"
version = "1.0"
release = version
copyright = "{}, Amazon.com".format(datetime.datetime.now().year)

extensions = ["sphinx.ext.autodoc", "sphinx.ext.viewcode", "sphinx.ext.napoleon", "sphinx.ext.todo"]

source_suffix = ".rst"
master_doc = "index"

autoclass_content = "class"
autodoc_member_order = "bysource"
default_role = "py:obj"

html_theme = "haiku"
htmlhelp_basename = "{}doc".format(project)

napoleon_use_rtype = False
