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

version = ag.load_version_file()
version = ag.update_version(version)

submodule = "core"
install_requires = (
    [
        # version ranges added in ag.get_dependency_version_ranges()
        "numpy",
        "scipy",
        "scikit-learn",
        "networkx",
        "pandas",
        "tqdm",
        "requests",
        "matplotlib",
        "boto3",
        f"autogluon.common=={version}",
        f"autogluon.features=={version}",
    ]
    if not ag.LITE_MODE
    else [
        # version ranges added in ag.get_dependency_version_ranges()
        "numpy",
        "scipy",
        "scikit-learn",
        "pandas",
        "tqdm",
        "matplotlib",
        f"{ag.PACKAGE_NAME}.common=={version}",
        f"{ag.PACKAGE_NAME}.features=={version}",
    ]
)


extras_require = {
    "ray": [
        # Ray<=2.53 does not support py313 on Windows
        "ray[default]>=2.43.0,<2.54; platform_system != 'Windows' or python_version != '3.13'",  # sync with common/src/autogluon/common/utils/try_import.py
    ],
    "raytune": [
        "pyarrow>=15.0.0",  # cap Pyarrow to fix source installation - https://github.com/autogluon/autogluon/issues/4519
        "ray[default,tune]>=2.43.0,<2.54; platform_system != 'Windows' or python_version != '3.13'",  # sync with common/src/autogluon/common/utils/try_import.py
        # TODO: consider alternatives as hyperopt is not actively maintained.
        "hyperopt>=0.2.7,<0.2.8",  # This is needed for the bayes search to work.
        # 'GPy>=1.10.0,<1.11.0'  # TODO: Enable this once PBT/PB2 are supported by ray lightning
        "stevedore<5.5",
        "setuptools<82",
    ],
}

tests_require = [
    "pytest",
    "types-requests",
    "types-setuptools",
    "pytest-mypy",
    "flake8",
    "pre-commit",
]

all_requires = []

for extra_package in ["ray", "raytune"]:
    if extra_package in extras_require:
        all_requires += extras_require[extra_package]
tests_require = list(set(tests_require))
all_requires = list(set(all_requires))
extras_require["tests"] = tests_require
extras_require["all"] = all_requires

install_requires = ag.get_dependency_version_ranges(install_requires)

if __name__ == "__main__":
    ag.create_version_file(version=version, submodule=submodule)
    setup_args = ag.default_setup_args(version=version, submodule=submodule)
    setup(
        install_requires=install_requires,
        extras_require=extras_require,
        **setup_args,
    )
