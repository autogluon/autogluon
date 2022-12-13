#!/usr/bin/env python
###########################
# This code block is a HACK (!), but is necessary to avoid code duplication. Do NOT alter these lines.
import importlib.util
import os

from setuptools import setup

filepath = os.path.abspath(os.path.dirname(__file__))
filepath_import = os.path.join(filepath, "..", "core", "src", "autogluon", "core", "_setup_utils.py")
spec = importlib.util.spec_from_file_location("ag_min_dependencies", filepath_import)
ag = importlib.util.module_from_spec(spec)
# Identical to `from autogluon.core import _setup_utils as ag`, but works without `autogluon.core` being installed.
spec.loader.exec_module(ag)
###########################

version = "0.1"
version = ag.update_version(version, use_file_if_exists=False, create_file=True)

submodule = "cloud"
install_requires = [
    # version ranges added in ag.get_dependency_version_ranges()
    "autogluon.common<0.7",
    "boto3",
    "numpy",
    "packaging<22",
    "pandas",
    # TODO: update to the latest after 0.6 container is out
    "sagemaker>=2.94",
    "pyarrow>=9.0",
    "Pillow>=9.3.0",
]

extras_require = dict()

test_requirements = [
    'tox',
    'pytest',
    'pytest-cov'
]

test_requirements = list(set(test_requirements))
extras_require['tests'] = test_requirements

install_requires = ag.get_dependency_version_ranges(install_requires)
for key in extras_require:
    extras_require[key] = ag.get_dependency_version_ranges(extras_require[key])

if __name__ == "__main__":
    ag.create_version_file(version=version, submodule=submodule)
    setup_args = ag.default_setup_args(version=version, submodule=submodule)
    setup(
        install_requires=install_requires,
        extras_require=extras_require,
        **setup_args,
    )
