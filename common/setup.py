#!/usr/bin/env python
# PEP 621 + uv workspace: pyproject.toml carries the static metadata + namespace
# config and declares version/dependencies/optional-dependencies as dynamic; this setup.py
# supplies those dynamic values at build time so published wheels keep their Requires-Dist:
#   * version -> base VERSION + a context suffix (single-sourced in _setup_utils: .dev0 for source
#                installs, bDATE for nightlies, none for releases), also written to version.py for
#                runtime `from .version import __version__`;
#   * deps    -> shared third-party caps from _setup_utils.DEPENDENT_PACKAGES (the single source
#                of truth for caps) + exact `==<version>` sibling pins.
# This is the documented canonical setup.py; the other submodules follow the same shape.
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
# Identical to `from autogluon.core import _setup_utils as ag`, but works without `autogluon.core` being installed.
spec.loader.exec_module(ag)
###########################

submodule = "common"
version = ag.update_version(ag.load_version_file())

install_requires = [
    # version ranges added in ag.get_dependency_version_ranges()
    "numpy",
    "pandas",
    "pyarrow",
    "boto3",
    "psutil",
    "tqdm",
    "requests",
    "joblib",
    "pyyaml",
    "packaging",
    # s3fs is removed due to doubling install time due to version range resolution
    # "s3fs",
    "scikit-learn",
]

extras_require = {
    "tests": [
        "pytest",
        "types-requests",
        "types-setuptools",
        "pytest-mypy",
    ],
}

install_requires = ag.get_dependency_version_ranges(install_requires)
extras_require = {key: ag.get_dependency_version_ranges(value) for key, value in extras_require.items()}

if __name__ == "__main__":
    ag.create_version_file(version=version, submodule=submodule)
    setup(
        version=version,
        long_description=ag.load_readme(),
        long_description_content_type="text/markdown",
        classifiers=ag.get_classifiers(),
        install_requires=install_requires,
        extras_require=extras_require,
    )
