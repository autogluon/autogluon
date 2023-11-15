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

import sys

version = ag.load_version_file()
version = ag.update_version(version)

submodule = "tabular"
install_requires = [
    # version ranges added in ag.get_dependency_version_ranges()
    "numpy",  # version range defined in `core/_setup_utils.py`
    "scipy",  # version range defined in `core/_setup_utils.py`
    "pandas",  # version range defined in `core/_setup_utils.py`
    "scikit-learn",  # version range defined in `core/_setup_utils.py`
    "networkx",  # version range defined in `core/_setup_utils.py`
    f"{ag.PACKAGE_NAME}.core=={version}",
    f"{ag.PACKAGE_NAME}.features=={version}",
]

extras_require = {
    "lightgbm": [
        "lightgbm>=3.3,<4.2",  # <{N+1} upper cap, where N is the latest released minor version
    ],
    "catboost": [
        # CatBoost wheel build is not working correctly on darwin for CatBoost 1.2, so use old version in this case.
        # https://github.com/autogluon/autogluon/pull/3190#issuecomment-1540599280
        # Catboost 1.2 doesn't have wheel for python 3.11
        "catboost>=1.1,<1.2 ; sys_platform == 'darwin' and python_version < '3.11'",
        "catboost>=1.1,<1.3; sys_platform != 'darwin'",
    ],
    # FIXME: Debug why xgboost 1.6 has 4x+ slower inference on multiclass datasets compared to 1.4
    #  It is possibly only present on MacOS, haven't tested linux.
    # XGBoost made API breaking changes in 1.6 with custom metric and callback support, so we don't support older versions.
    "xgboost": [
        "xgboost>=1.6,<1.8",  # Holding the old version, 2.0 - major release breaks tests
    ],
    "fastai": [
        "torch",  # version range defined in `core/_setup_utils.py`
        "fastai>=2.3.1,<2.8",
    ],
    "tabpfn": [
        "tabpfn>=0.1,<0.2",
    ],
    "ray": [
        f"{ag.PACKAGE_NAME}.core[all]=={version}",
    ],
    "skex": [
        # Note: 2021.7 released on Sep 2022, version 2022.x doesn't exist (went directly from 2021.7 to 2023.0)
        "scikit-learn-intelex>=2021.7,<2023.2",
    ],
    "imodels": [
        "imodels>=1.3.10,<1.4.0",  # 1.3.8/1.3.9 either remove/renamed attribute `complexity_` causing failures. https://github.com/csinva/imodels/issues/147
    ],
    "vowpalwabbit": [
        # FIXME: 9.5+ causes VW to save an empty model which always predicts 0. Confirmed on MacOS (Intel CPU). Unknown how to fix.
        # No vowpalwabbit wheel for python 3.11 or above yet
        "vowpalwabbit>=9,<9.9; python_version < '3.11'",
    ],
    "skl2onnx": [
        "skl2onnx>=1.15.0,<1.16.0",
        # For macOS, there isn't a onnxruntime-gpu package installed with skl2onnx.
        # Therefore, we install onnxruntime explicitly here just for macOS.
        "onnxruntime>=1.15.0,<1.16.0",
    ]
    if sys.platform == "darwin"
    else ["skl2onnx>=1.15.0,<1.16.0", "onnxruntime-gpu>=1.15.0,<1.16.0"],
}

# TODO: v1.0: Rename `all` to `core`, make `all` contain everything.
all_requires = []
# TODO: Consider adding 'skex' to 'all'
for extra_package in ["lightgbm", "catboost", "xgboost", "fastai", "ray"]:
    all_requires += extras_require[extra_package]
all_requires = list(set(all_requires))
extras_require["all"] = all_requires


test_requires = []
for test_package in ["tabpfn", "imodels", "vowpalwabbit", "skl2onnx"]:
    test_requires += extras_require[test_package]
extras_require["tests"] = test_requires
install_requires = ag.get_dependency_version_ranges(install_requires)
extras_require = {key: ag.get_dependency_version_ranges(value) for key, value in extras_require.items()}

if __name__ == "__main__":
    ag.create_version_file(version=version, submodule=submodule)
    setup_args = ag.default_setup_args(version=version, submodule=submodule)
    setup(
        install_requires=install_requires,
        extras_require=extras_require,
        **setup_args,
    )
