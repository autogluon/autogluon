#!/usr/bin/env python
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
ag = importlib.util.module_from_spec(spec)  # type: ignore
# Identical to `from autogluon.core import _setup_utils as ag`, but works without `autogluon.core` being installed.
spec.loader.exec_module(ag)  # type: ignore
###########################

version = ag.load_version_file()
version = ag.update_version(version)

submodule = "eda"
install_requires = [
    # version ranges added in ag.get_dependency_version_ranges()
    "numpy",  # version range defined in `core/_setup_utils.py`
    "scipy",  # version range defined in `core/_setup_utils.py`
    "scikit-learn",  # version range defined in `core/_setup_utils.py`
    "pandas",  # version range defined in `core/_setup_utils.py`
    "matplotlib",  # version range defined in `core/_setup_utils.py`
    "missingno>=0.5.1,<0.6",
    "phik>=0.12.2,<0.13",
    "seaborn>=0.12.0,<0.14",
    "ipywidgets>=7.7.1,<9.0",  # min versions guidance: 7.7.1 collab/kaggle
    "shap>=0.44,<0.47",
    "yellowbrick>=1.5,<1.6",
    "pyod>=1.1,<1.2",
    "suod>=0.0.8,<0.1",
    "ipython>7.16,<8.13",  # IPython 8.13+ supports Python 3.9 and above; Python 3.7 is supported with IPython >7.16
    f"autogluon.core=={version}",
    f"autogluon.common=={version}",
    f"autogluon.features=={version}",
    f"autogluon.tabular=={version}",
]

extras_require = dict()

test_requirements = [
    "tox",
    "pytest",
    "pytest-cov",
    "types-requests",
    "types-setuptools",
    "pytest-mypy",
    "PyHamcrest",
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
