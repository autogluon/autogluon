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
    "requests>=2.21,<3",
    "jsonschema<4.18",
    "seqeval<1.3.0",
    "evaluate<0.4.0",
    "accelerate>=0.9,<0.17",
    "timm<0.7.0",
    "torch>=1.9,<1.14",
    "torchvision<0.15.0",
    "fairscale>=0.4.5,<0.4.14",
    "scikit-image>=0.19.1,<0.20.0",
    "pytorch-lightning>=1.8.0,<1.10.0",
    "text-unidecode<1.4",
    "torchmetrics>=0.8.0,<0.9.0",
    "transformers>=4.23.0,<4.27.0",
    "nptyping>=1.4.4,<2.5.0",
    "omegaconf>=2.1.1,<2.3.0",
    "sentencepiece>=0.1.95,<0.2.0",
    f"autogluon.core[raytune]=={version}",
    f"autogluon.features=={version}",
    f"autogluon.common=={version}",
    "pytorch-metric-learning>=1.3.0,<2.0",
    "nlpaug>=1.1.10,<1.2.0",
    "nltk>=3.4.5,<4.0.0",
    "openmim>0.1.5,<0.4.0",
    "defusedxml>=0.7.1,<0.7.2",
    "jinja2>=3.0.3,<3.2",
    "tensorboard",
    "pytesseract<0.3.11",
]

install_requires = ag.get_dependency_version_ranges(install_requires)

extras_require = {
    "tests": [
        "black>=22.3,<23.0",
        "isort>=5.10",
        "datasets>=2.3.2,<=2.3.2",
        "onnx>=1.13.0,<1.14.0",
        "onnxruntime>=1.13.0,<1.14.0;platform_system=='Darwin'",
        "onnxruntime-gpu>=1.13.0,<1.14.0;platform_system!='Darwin'",
    ]
}


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
