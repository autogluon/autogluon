#!/usr/bin/env python
# Thin setup.py: see common/setup.py. Only supplies the dynamic version + writes version.py.
# This is the meta package (submodule=None).
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

submodule = None  # meta package
version = ag.update_version(ag.load_version_file())

if __name__ == "__main__":
    ag.create_version_file(version=version, submodule=submodule)
    setup(version=version)
