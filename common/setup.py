#!/usr/bin/env python
# Thin setup.py: all PEP 621 metadata lives in pyproject.toml. This only supplies the dynamic
# version (base VERSION + optional AUTOGLUON_VERSION_SUFFIX, single-sourced) and writes version.py
# for runtime `from .version import __version__`. Kept so the nightly `setup.py sdist` path and
# the single-source version model are preserved; could later be removed via a build-backend
# version plugin for pure PEP 621.
###########################
# This code block is a HACK (!), but is necessary to avoid code duplication. Do NOT alter these lines.
import importlib.util
import os

from setuptools import setup

filepath = os.path.abspath(os.path.dirname(__file__))
filepath_import = os.path.join(filepath, "..", "core", "src", "autogluon", "core", "_setup_utils.py")
if not os.path.exists(filepath_import):
    filepath_import = os.path.join(filepath, "_setup_utils.py")
spec = importlib.util.spec_from_file_location("ag_min_dependencies", filepath_import)
ag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ag)
###########################

submodule = "common"
version = ag.update_version(ag.load_version_file())

if __name__ == "__main__":
    ag.create_version_file(version=version, submodule=submodule)
    setup(version=version)
