#!/usr/bin/env python
###########################
# This code block is a HACK (!), but is necessary to avoid code duplication. Do NOT alter these lines.
import importlib.util
import os

from setuptools import setup

filepath = os.path.abspath(os.path.dirname(__file__))
filepath_import = os.path.join(filepath, "..", "core", "src", "autogluon", "core", "_setup_utils.py")
spec = importlib.util.spec_from_file_location("ag_min_dependencies", filepath_import)
ag = importlib.util.module_from_spec(spec)  # type: ignore
# Identical to `from autogluon.core import _setup_utils as ag`, but works without `autogluon.core` being installed.
spec.loader.exec_module(ag)  # type: ignore
###########################

version = ag.load_version_file()
version = ag.update_version(version, use_file_if_exists=False, create_file=True)

submodule = "common"
install_requires = (
    [
        # version ranges added in ag.get_dependency_version_ranges()
        "numpy",  # version range defined in `core/_setup_utils.py`
        "pandas",  # version range defined in `core/_setup_utils.py`
        "pyarrow",  # version range defined in `core/_setup_utils.py`
        "boto3",  # version range defined in `core/_setup_utils.py`
        "psutil",  # version range defined in `core/_setup_utils.py`
        "tqdm",  # version range defined in `core/_setup_utils.py`
        "requests",
        # s3fs is removed due to doubling install time due to version range resolution
        # "s3fs",  # version range defined in `core/_setup_utils.py`
    ]
    if not ag.LITE_MODE
    else {
        "numpy",  # version range defined in `core/_setup_utils.py`
        "pandas",  # version range defined in `core/_setup_utils.py`
        "tqdm",  # version range defined in `core/_setup_utils.py`
        "requests",
    }
)

extras_require = dict()

test_requirements = [
    "pytest",
    "types-requests",
    "types-setuptools",
    "pytest-mypy",
]

test_requirements = list(set(test_requirements))
extras_require["tests"] = test_requirements

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
