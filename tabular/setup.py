#!/usr/bin/env python
# Thin setup.py: see common/setup.py. Supplies the dynamic version + computed deps
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

submodule = "tabular"
version = ag.update_version(ag.load_version_file())

install_requires = [
    # version ranges added in ag.get_dependency_version_ranges()
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "networkx",
    f"autogluon.core=={version}",
    f"autogluon.features=={version}",
]

extras_require = {
    "lightgbm": [
        "lightgbm>=4.0,<4.8",  # <{N+1} upper cap, where N is the latest released minor version
    ],
    "catboost": [
        "catboost>=1.2,<1.3",
    ],
    "xgboost": [
        # The CPU-only build. The default `xgboost` wheel depends on `nvidia-nccl-cu12` while
        # torch depends on `nvidia-nccl-cu13`, and both wheels ship
        # `nvidia/nccl/lib/libnccl.so.2`. With both installed, the NCCL that torch loads is
        # whichever wheel wrote that path last rather than the one torch pins, and uninstalling
        # either wheel deletes the file the other still needs. Observed on a fresh
        # `[tabarena]` install: nccl-cu12 2.27.5 landed last and torch 2.13 then failed to
        # import with `undefined symbol: ncclCommResume`. `xgboost-cpu` has no NCCL dependency,
        # which removes the overlap. Trade-off: it is built without CUDA, so XGBoost trains on
        # CPU even when GPUs are allocated (it warns and falls back); no GPU-targeted portfolio
        # contains an XGB config. Revisit if xgboost ships a cu13 wheel or makes NCCL optional
        # (https://github.com/dmlc/xgboost/issues/10729).
        "xgboost-cpu>=2.1.1,<3.4",  # >=2.1.1 is the earliest xgboost-cpu release; <{N+1} upper cap
    ],
    "realmlp": [
        "pytabkit>=1.7.2,<1.8",
    ],
    "interpret": [
        "interpret-core>=0.7.2,<0.8",
    ],
    "fastai": [
        "spacy<3.9",
        "torch",  # version range defined in `core/_setup_utils.py`
        # Held below 2.8.8: that release requires fastcore>=1.14.6, but the highest fastcore 1.x
        # is 1.14.5, so it is unreachable under the fastcore<2 cap below. Lifting that cap needs
        # the removed L.starmap usage replaced first.
        "fastai>=2.3.1,<2.8.8",  # Cap for major version
        "fastcore<2",  # Breaking change in v2: removed L.starmap, which breaks fastai models
    ],
    "tabm": [
        "torch",  # version range defined in `core/_setup_utils.py`
    ],
    "tabpfn": [
        "tabpfn>=8.0,<8.3",  # <{N+1} upper cap, where N is the latest released minor version; >=8.0 for the TabPFN-3 checkpoints
    ],
    "tabdpt": [
        "tabdpt>=1.2,<1.3",  # >=1.2 for TabDPT-Turbo; v1.1 weights stay pinned per model class
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
        "tabicl>=2.0,<2.2",  # <{N+1} upper cap, where N is the latest released minor version
    ],
    "nori": [
        "synthefy-nori>=0.13,<0.15",  # <{N+1} upper cap, where N is the latest released minor version
    ],
    "ray": [
        f"autogluon.core[all]=={version}",
    ],
    "skex": [
        "scikit-learn-intelex>=2025.0,<2026.2",  # <{N+1} upper cap, where N is the latest released minor version
    ],
    "imodels": [
        "imodels>=1.3.10,<2.1.0",  # 1.3.8/1.3.9 either remove/renamed attribute `complexity_` causing failures. https://github.com/csinva/imodels/issues/147
    ],
    "skl2onnx": [
        "skl2onnx>=1.20.0,<1.21.0",
        # Sync ONNX requirements with multimodal/setup.py
        "onnx>=1.21.0,<1.23.0",
        # For macOS, there isn't a onnxruntime-gpu package installed with skl2onnx.
        # Therefore, we install onnxruntime explicitly here just for macOS.
        "onnxruntime>=1.17.0,<1.24.0",
        "onnxruntime-gpu>=1.17.0,<1.24.0; platform_system != 'Darwin' and platform_machine != 'aarch64'",
    ],
}

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

tabarena_requires = list(all_requires)
for extra_package in [
    "tabdpt",
    "tabicl",
    "tabpfn",
    "realmlp",
    "nori",
]:
    tabarena_requires += extras_require[extra_package]
tabarena_requires = list(set(tabarena_requires))
extras_require["tabarena"] = tabarena_requires

test_requires = []
for test_package in [
    "interpret",
    "tabdpt",
    "tabicl",  # Currently has unnecessary extra dependencies such as xgboost and wandb
    "tabpfn",
    "realmlp",  # Will consider to put as part of `all_requires` once part of a portfolio
    "tabpfnmix",  # Refer to `mitra`, which is an improved version of `tabpfnmix`
    "nori",
    "imodels",
    "skl2onnx",
]:
    test_requires += extras_require[test_package]
extras_require["tests"] = test_requires

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
