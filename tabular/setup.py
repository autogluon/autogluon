#!/usr/bin/env python
###########################
# This code block is a HACK (!), but is necessary to avoid code duplication. Do NOT alter these lines.
import importlib.util
import os
import platform

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
        "lightgbm>=4.0,<4.6",  # <{N+1} upper cap, where N is the latest released minor version
    ],
    "catboost": [
        "numpy>=1.25,<2.0.0",  # TODO support numpy>=2.0.0 once issue resolved https://github.com/catboost/catboost/issues/2671
        "catboost>=1.2,<1.3",
    ],
    # FIXME: Debug why xgboost 1.6 has 4x+ slower inference on multiclass datasets compared to 1.4
    #  It is possibly only present on MacOS, haven't tested linux.
    # XGBoost made API breaking changes in 1.6 with custom metric and callback support, so we don't support older versions.
    "xgboost": [
        "xgboost>=1.6,<2.2",  # <{N+1} upper cap, where N is the latest released minor version
    ],
    "fastai": [
        "spacy<3.8",  # cap for issue https://github.com/explosion/spaCy/issues/13653
        "torch",  # version range defined in `core/_setup_utils.py`
        "fastai>=2.3.1,<2.8",  # <{N+1} upper cap, where N is the latest released minor version
    ],
    "tabpfn": [
        "tabpfn>=0.1,<0.2",  # <{N+1} upper cap, where N is the latest released minor version
    ],
    "tabpfnmix": [
        "torch",  # version range defined in `core/_setup_utils.py`
        "huggingface_hub[torch]",  # Only needed for HuggingFace downloads, currently uncapped to minimize future conflicts.
        "einops>=0.7,<0.9",
    ],
    "ray": [
        f"{ag.PACKAGE_NAME}.core[all]=={version}",
    ],
    "skex": [
        "scikit-learn-intelex>=2024.0,<2025.1",  # <{N+1} upper cap, where N is the latest released minor version
    ],
    "imodels": [
        "imodels>=1.3.10,<1.4.0",  # 1.3.8/1.3.9 either remove/renamed attribute `complexity_` causing failures. https://github.com/csinva/imodels/issues/147
    ],
    "vowpalwabbit": [
        # FIXME: 9.5+ causes VW to save an empty model which always predicts 0. Confirmed on MacOS (Intel CPU). Unknown how to fix.
        # No vowpalwabbit wheel for python 3.11 or above yet
        "vowpalwabbit>=9,<9.10; python_version < '3.11' and sys_platform != 'darwin'",
    ],
    
}

is_aarch64 = platform.machine() == 'aarch64'
is_darwin = sys.platform == 'darwin'

if is_darwin or is_aarch64:
# For macOS or aarch64, only use CPU version
    extras_require["skl2onnx"] = [
        "onnx>=1.13.0,<1.16.2;platform_system=='Windows'",  # cap at 1.16.1 for issue https://github.com/onnx/onnx/issues/6267
        "onnx>=1.13.0,<1.18.0;platform_system!='Windows'",
        "skl2onnx>=1.15.0,<1.18.0",
        # For macOS, there isn't a onnxruntime-gpu package installed with skl2onnx.
        # Therefore, we install onnxruntime explicitly here just for macOS.
        "onnxruntime>=1.17.0,<1.20.0",
    ]
else:
# For other platforms, include both CPU and GPU versions
    extras_require["skl2onnx"] = [
        "onnx>=1.13.0,<1.16.2;platform_system=='Windows'",  # cap at 1.16.1 for issue https://github.com/onnx/onnx/issues/6267
        "onnx>=1.13.0,<1.18.0;platform_system!='Windows'",
        "skl2onnx>=1.15.0,<1.18.0", 
        "onnxruntime>=1.17.0,<1.20.0",   # install for gpu system due to https://github.com/autogluon/autogluon/issues/3804
        "onnxruntime-gpu>=1.17.0,<1.20.0",
    ]

# TODO: v1.0: Rename `all` to `core`, make `all` contain everything.
all_requires = []
# TODO: Consider adding 'skex' to 'all'
for extra_package in ["lightgbm", "catboost", "xgboost", "fastai", "tabpfnmix", "ray"]:
    all_requires += extras_require[extra_package]
all_requires = list(set(all_requires))
extras_require["all"] = all_requires


test_requires = []
for test_package in ["tabpfn", "tabpfnmix", "imodels", "vowpalwabbit", "skl2onnx"]:
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
