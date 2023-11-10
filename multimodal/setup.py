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
    "requests>=2.21,<3",
    "jsonschema>=4.14,<4.18",
    "seqeval>=1.2.2,<1.3.0",
    "evaluate>=0.4.0,<0.5.0",
    "accelerate>=0.21.0,<0.22.0",
    "transformers[sentencepiece]>=4.31.0,<4.32.0",
    "timm>=0.9.5,<0.10.0",
    "torchvision>=0.14.0,<0.16.0",  # torch 1.13 requires torchvision 0.14. Increase it to 0.15 when dropping the support of torch 1.13.
    "scikit-image>=0.19.1,<0.21.0",
    "text-unidecode>=1.3,<1.4",
    "torchmetrics>=1.0.0,<1.2.0",
    "nptyping>=1.4.4,<2.5.0",
    "omegaconf>=2.1.1,<2.3.0",
    f"autogluon.core[raytune]=={version}",
    f"autogluon.features=={version}",
    f"autogluon.common=={version}",
    "pytorch-metric-learning>=1.3.0,<2.0",
    "nlpaug>=1.1.10,<1.2.0",
    "nltk>=3.4.5,<4.0.0",
    "openmim>=0.3.7,<0.4.0",
    "defusedxml>=0.7.1,<0.7.2",
    "jinja2>=3.0.3,<3.2",
    "tensorboard>=2.9,<3",
    "pytesseract>=0.3.9,<0.3.11",
    "nvidia-ml-py3==7.352.0",
]

install_requires = ag.get_dependency_version_ranges(install_requires)

tests_require = [
    "black~=23.0",
    "isort>=5.10",
    "datasets>=2.10.0,<2.15.0",
    "onnx>=1.13.0,<1.14.0",
    "onnxruntime>=1.15.0,<1.16.0;platform_system=='Darwin'",
    "onnxruntime-gpu>=1.15.0,<1.16.0;platform_system!='Darwin'",
    "tensorrt>=8.5.3.1,<8.5.4;platform_system=='Linux' and python_version<'3.11'",  # tensorrt > 8.5.4 cause segfault
]

extras_require = {
    "PyMuPDF": [
        "PyMuPDF<=1.21.1",
    ]
}

for test_package in ["PyMuPDF"]:
    tests_require += extras_require[test_package]

extras_require["tests"] = tests_require


if __name__ == "__main__":
    ag.create_version_file(version=version, submodule=submodule)
    setup_args = ag.default_setup_args(version=version, submodule=submodule)
    setup_args["package_data"]["autogluon.multimodal"] = [
        "configs/data/*.yaml",
        "configs/model/*.yaml",
        "configs/optimization/*.yaml",
        "configs/environment/*.yaml",
        "configs/distiller/*.yaml",
        "configs/matcher/*.yaml",
    ]
    setup(
        install_requires=install_requires,
        extras_require=extras_require,
        **setup_args,
    )
