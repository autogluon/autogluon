#!/usr/bin/env python
# Option B thin setup.py: see common/setup.py. Supplies the dynamic version + computed deps
# (caps from _setup_utils.DEPENDENT_PACKAGES + exact `==<version>` sibling pins) and writes version.py.
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

submodule = "core"
version = ag.update_version(ag.load_version_file())

install_requires = [
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
    "typing-extensions",
    f"autogluon.common=={version}",
    f"autogluon.features=={version}",
]

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
    all_requires += extras_require[extra_package]
all_requires = list(set(all_requires))
extras_require["all"] = all_requires
extras_require["tests"] = list(set(tests_require))

install_requires = ag.get_dependency_version_ranges(install_requires)

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
