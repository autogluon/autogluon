#!/usr/bin/env python
import importlib.util

###########################
# This code block is a HACK (!), but is necessary to avoid code duplication. Do NOT alter these lines.
import os
import warnings

from packaging.version import parse as vparse
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

submodule = "timeseries"
install_requires = [
    # version ranges added in ag.get_dependency_version_ranges()
    "joblib>=1.1,<2",
    "numpy",  # version range defined in `core/_setup_utils.py`
    "scipy",  # version range defined in `core/_setup_utils.py`
    "pandas",  # version range defined in `core/_setup_utils.py`
    "statsmodels>=0.13.0,<0.14",
    "gluonts>=0.12.0,<0.13",
    "torch>=1.9,<1.14",
    "pytorch-lightning>=1.7.4,<1.10.0",
    "networkx",  # version range defined in `core/_setup_utils.py`
    "statsforecast>=1.4.0,<1.5",
    "tqdm",  # version range defined in `core/_setup_utils.py`
    "ujson>=5,<6",  # needed to silence GluonTS warning
    f"autogluon.core[raytune]=={version}",
    f"autogluon.common=={version}",
    f"autogluon.tabular[catboost,lightgbm,xgboost]=={version}",
]

extras_require = {
    "tests": [
        "pytest",
        "flake8>=4.0,<5",
        "flaky>=3.7,<4",
        "pytest-timeout>=2.1,<3",
        "isort>=5.10",
        "black>=22.3,<23.0",
    ],
    "sktime": ["sktime>=0.14,<0.16", "pmdarima>=1.8.2,<1.9", "tbats>=1.1,<2"],
}

all_requires = []
for extra_package in ["sktime"]:
    all_requires += extras_require[extra_package]
extras_require["all"] = list(set(all_requires))

install_requires = ag.get_dependency_version_ranges(install_requires)

if __name__ == "__main__":
    ag.create_version_file(version=version, submodule=submodule)
    setup_args = ag.default_setup_args(version=version, submodule=submodule)
    setup(
        install_requires=install_requires,
        extras_require=extras_require,
        **setup_args,
    )
