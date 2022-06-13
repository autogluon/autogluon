#!/usr/bin/env python
###########################
# This code block is a HACK (!), but is necessary to avoid code duplication. Do NOT alter these lines.
import os
from setuptools import setup
import importlib.util

filepath = os.path.abspath(os.path.dirname(__file__))
filepath_import = os.path.join(
    filepath, "..", "core", "src", "autogluon", "core", "_setup_utils.py"
)
spec = importlib.util.spec_from_file_location("ag_min_dependencies", filepath_import)
ag = importlib.util.module_from_spec(spec)
# Identical to `from autogluon.core import _setup_utils as ag`, but works without `autogluon.core` being installed.
spec.loader.exec_module(ag)
###########################

version = ag.load_version_file()
version = ag.update_version(version)

submodule = "timeseries"
install_requires = [
    # version ranges added in ag.get_dependency_version_ranges()
    "numpy",
    "scipy",
    "pandas",
    "psutil>=5.7.3,<5.9",
    "gluonts>=0.8.0",
    "mxnet<2.0",
    f"autogluon.core=={version}",
]

extras_require = {
    "tests": ["pytest", "flake8", "flaky", "pytest-timeout"]
}
install_requires = ag.get_dependency_version_ranges(install_requires)

if __name__ == "__main__":
    ag.create_version_file(version=version, submodule=submodule)
    setup_args = ag.default_setup_args(version=version, submodule=submodule)
    setup(
        install_requires=install_requires,
        extras_require=extras_require,
        **setup_args,
    )
