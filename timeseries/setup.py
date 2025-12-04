#!/usr/bin/env python
import importlib.util

###########################
# This code block is a HACK (!), but is necessary to avoid code duplication. Do NOT alter these lines.
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

version = ag.load_version_file()
version = ag.update_version(version)

submodule = "timeseries"
install_requires = [
    # version ranges added in ag.get_dependency_version_ranges()
    "joblib",  # version range defined in `core/_setup_utils.py`
    "numpy",  # version range defined in `core/_setup_utils.py`
    "scipy",  # version range defined in `core/_setup_utils.py`
    "pandas",  # version range defined in `core/_setup_utils.py`
    "torch",  # version range defined in `core/_setup_utils.py`
    "lightning",  # version range defined in `core/_setup_utils.py`
    "transformers[sentencepiece]",  # version range defined in `core/_setup_utils.py`
    "accelerate",  # version range defined in `core/_setup_utils.py`
    "gluonts>=0.15.0,<0.17",
    "networkx",  # version range defined in `core/_setup_utils.py`
    "statsforecast>=1.7.0,<2.0.2",
    "mlforecast>=0.14.0,<0.15.0",  # cannot upgrade since v0.15.0 introduced a breaking change to DirectTabular
    "utilsforecast>=0.2.3,<0.2.12",  # to prevent breaking changes that propagate through mlforecast's dependency
    "coreforecast>=0.0.12,<0.0.17",  # to prevent breaking changes that propagate through mlforecast's dependency
    "fugue>=0.9.0",  # prevent dependency clash with omegaconf
    "tqdm",  # version range defined in `core/_setup_utils.py`
    "orjson~=3.9",  # use faster JSON implementation in GluonTS
    "einops>=0.7,<1",  # required by Chronos-2 and Toto
    "chronos-forecasting>=2.2.0rc4,<3",
    "peft>=0.13.0,<0.18",  # version range same as in chronos-forecasting[extras]
    "tensorboard>=2.9,<3",  # fixes https://github.com/autogluon/autogluon/issues/3612
    f"autogluon.core=={version}",
    f"autogluon.common=={version}",
    f"autogluon.features=={version}",
    f"autogluon.tabular[catboost,lightgbm,xgboost]=={version}",
]

extras_require = {
    "tests": [
        "pytest",
        "ruff>=0.0.285",
        "flaky>=3.7,<4",
        "pytest-timeout>=2.1,<3",
    ],
    "ray": [
        f"autogluon.core[raytune]=={version}",
    ],
}

extras_require["all"] = list(set.union(*(set(extras_require[extra]) for extra in ["ray"])))
install_requires = ag.get_dependency_version_ranges(install_requires)

if __name__ == "__main__":
    ag.create_version_file(version=version, submodule=submodule)
    setup_args = ag.default_setup_args(version=version, submodule=submodule)
    setup(
        install_requires=install_requires,
        extras_require=extras_require,
        **setup_args,
    )
