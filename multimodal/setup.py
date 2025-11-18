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

version = ag.load_version_file()
version = ag.update_version(version)

submodule = "multimodal"
install_requires = [
    # version ranges added in ag.get_dependency_version_ranges()
    "numpy",  # version range defined in `core/_setup_utils.py`
    "scipy",  # version range defined in `core/_setup_utils.py`
    "pandas",  # version range defined in `core/_setup_utils.py`
    "scikit-learn",  # version range defined in `core/_setup_utils.py`
    "Pillow",  # version range defined in `core/_setup_utils.py`
    "tqdm",  # version range defined in `core/_setup_utils.py`
    "boto3",  # version range defined in `core/_setup_utils.py`
    "torch",  # version range defined in `core/_setup_utils.py`
    "lightning",  # version range defined in `core/_setup_utils.py`
    "transformers[sentencepiece]",  # version range defined in `core/_setup_utils.py`
    "accelerate",  # version range defined in `core/_setup_utils.py`
    "fsspec[http]<=2025.3",  # pin version to avoid conflicts with `datasets`
    "requests>=2.30,<3",
    "jsonschema>=4.18,<4.24",
    "seqeval>=1.2.2,<1.3.0",
    "evaluate>=0.4.0,<0.5.0",
    "timm>=0.9.5,<1.0.7",
    "torchvision>=0.21.0,<0.23.0",
    "scikit-image>=0.19.1,<0.26.0",
    "text-unidecode>=1.3,<1.4",
    "torchmetrics>=1.2.0,<1.8",
    "omegaconf>=2.1.1,<2.4.0",
    f"autogluon.core[raytune]=={version}",
    f"autogluon.features=={version}",
    f"autogluon.common=={version}",
    "pytorch-metric-learning>=1.3.0,<2.9",
    "nlpaug>=1.1.10,<1.2.0",
    "nltk>=3.4.5,<3.10",  # Updated upper bound to address CVE-2024-39705
    "openmim>=0.3.7,<0.4.0",
    "defusedxml>=0.7.1,<0.7.2",
    "jinja2>=3.0.3,<3.2",
    "tensorboard>=2.9,<3",
    "pytesseract>=0.3.9,<0.4",
    "nvidia-ml-py3>=7.352.0, <8.0",
    "pdf2image>=1.17.0,<1.19",
]

install_requires = ag.get_dependency_version_ranges(install_requires)

tests_require = [
    "ruff",
    "datasets>=2.16.0,<3.6.0",
    "onnx>=1.13.0,<1.16.2;platform_system=='Windows'",  # cap at 1.16.1 for issue https://github.com/onnx/onnx/issues/6267
    "onnx>=1.13.0,<1.18.0;platform_system!='Windows'",
    "onnxruntime>=1.17.0,<1.22.0",  # install for gpu system due to https://github.com/autogluon/autogluon/issues/3804
    "onnxruntime-gpu>=1.17.0,<1.22.0;platform_system!='Darwin' and platform_machine!='aarch64'",
    "tensorrt>=8.6.0,<10.9.1;platform_system=='Linux' and python_version<'3.11'",
]

extras_require = {"tests": tests_require}


if __name__ == "__main__":
    ag.create_version_file(version=version, submodule=submodule)
    setup_args = ag.default_setup_args(version=version, submodule=submodule)
    setup_args["package_data"]["autogluon.multimodal"] = [
        "configs/data/*.yaml",
        "configs/model/*.yaml",
        "configs/optim/*.yaml",
        "configs/env/*.yaml",
        "configs/distiller/*.yaml",
        "configs/matcher/*.yaml",
    ]
    setup(
        install_requires=install_requires,
        extras_require=extras_require,
        **setup_args,
    )
