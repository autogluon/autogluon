#!/usr/bin/env python
###########################
# This code block is a HACK (!), but is necessary to avoid code duplication. Do NOT alter these lines.
import importlib.util
import os
import platform

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

import copy
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
        "lightgbm>=4.0,<4.7",  # <{N+1} upper cap, where N is the latest released minor version
    ],
    "catboost": [
        "numpy>=1.25,<2.3.0",
        "catboost>=1.2,<1.3",
    ],
    "xgboost": [
        "xgboost>=2.0,<3.1",  # <{N+1} upper cap, where N is the latest released minor version
    ],
    "realmlp": [
        "pytabkit>=1.6,<1.7",
    ],
    "interpret": [
        "interpret-core>=0.7.2,<0.8",
    ],
    "fastai": [
        "spacy<3.9",
        "torch",  # version range defined in `core/_setup_utils.py`
        "fastai>=2.3.1,<2.9",  # <{N+1} upper cap, where N is the latest released minor version
    ],
    "tabm": [
        "torch",  # version range defined in `core/_setup_utils.py`
    ],
    "tabpfn": [
        "tabpfn>=2.0.9,<2.2",  # <{N+1} upper cap, where N is the latest released minor version
    ],
    "tabpfnmix": [
        "torch",  # version range defined in `core/_setup_utils.py`
        "huggingface_hub[torch]",  # version range defined in `core/_setup_utils.py`
        "einops>=0.7,<0.9",
    ],
    "mitra": [
        "loguru",
        "einx",
        "omegaconf",
        "torch",
        "transformers",
        "huggingface_hub[torch]",  # version range defined in `core/_setup_utils.py`
        "einops>=0.7,<0.9",
    ],
    "tabicl": [
        "tabicl>=0.1.3,<0.2",  # 0.1.3 added a major bug fix to multithreading.
    ],
    "ray": [
        f"{ag.PACKAGE_NAME}.core[all]=={version}",
    ],
    "skex": [
        "scikit-learn-intelex>=2024.0,<2025.5",  # <{N+1} upper cap, where N is the latest released minor version
    ],
    "imodels": [
        "imodels>=1.3.10,<2.1.0",  # 1.3.8/1.3.9 either remove/renamed attribute `complexity_` causing failures. https://github.com/csinva/imodels/issues/147
    ],
}

extras_require["skl2onnx"] = [
    "skl2onnx>=1.15.0,<1.20.0",
    # Sync ONNX requirements with multimodal/setup.py
    "onnx>=1.13.0,!=1.16.2,<1.21.0;platform_system=='Windows'",  # exclude 1.16.1 for issue https://github.com/onnx/onnx/issues/6267
    "onnx>=1.13.0,<1.21.0;platform_system!='Windows'",
    # For macOS, there isn't a onnxruntime-gpu package installed with skl2onnx.
    # Therefore, we install onnxruntime explicitly here just for macOS.
    "onnxruntime>=1.17.0,<1.24.0",
    "onnxruntime-gpu>=1.17.0,<1.24.0; platform_system != 'Darwin' and platform_machine != 'aarch64'",
]

# TODO: v1.0: Rename `all` to `core`, make `all` contain everything.
all_requires = []
for extra_package in [
    "lightgbm",
    "catboost",
    "xgboost",
    "fastai",
    "tabm",
    "mitra",
    "ray",
]:
    all_requires += extras_require[extra_package]
all_requires = list(set(all_requires))
extras_require["all"] = all_requires

tabarena_requires = copy.deepcopy(all_requires)
for extra_package in [
    "interpret",
    "tabicl",
    "tabpfn",
    "realmlp",
]:
    tabarena_requires += extras_require[extra_package]
tabarena_requires = list(set(tabarena_requires))
extras_require["tabarena"] = tabarena_requires

test_requires = []
for test_package in [
    "interpret",
    "tabicl",  # Currently has unnecessary extra dependencies such as xgboost and wandb
    "tabpfn",
    "realmlp",  # Will consider to put as part of `all_requires` once part of a portfolio
    "tabpfnmix",  # Refer to `mitra`, which is an improved version of `tabpfnmix`
    "imodels",
    "skl2onnx",
]:
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
