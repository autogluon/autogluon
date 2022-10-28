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
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "Pillow",
    "tqdm",
    "boto3",
    "requests",
    "jsonschema<=4.8.0",
    "seqeval<=1.2.2",
    "evaluate<=0.2.2",
    "timm<0.7.0",
    "torch>=1.9,<1.13",
    "torchvision<0.14.0",
    "torchtext<0.14.0",
    "fairscale>=0.4.5,<=0.4.6",
    "scikit-image>=0.19.1,<0.20.0",
    "smart_open>=5.2.1,<5.3.0",
    "pytorch_lightning>=1.7.4,<1.8.0",
    "text-unidecode<=1.3",
    "torchmetrics>=0.8.0,<0.9.0",
    "transformers>=4.23.0,<4.24.0",
    "nptyping>=1.4.4,<1.5.0",
    "omegaconf>=2.1.1,<2.2.0",
    "sentencepiece>=0.1.95,<0.2.0",
    f"autogluon.core[raytune]=={version}",
    f"autogluon.features=={version}",
    f"autogluon.common=={version}",
    "pytorch-metric-learning>=1.3.0,<1.4.0",
    "nlpaug>=1.1.10,<=1.1.10",
    "nltk>=3.4.5,<4.0.0",
    "openmim>0.1.5,<=0.2.1",
    "pycocotools>=2.0.4,<=2.0.4",
    "defusedxml>=0.7.1,<=0.7.1",
    "pytrec-eval>=0.5,<=0.5",
    "albumentations>=1.1.0,<=1.2.0",
]

install_requires = ag.get_dependency_version_ranges(install_requires)

extras_require = {
    "tests": [
        "black~=22.0,>=22.3",
        "isort>=5.10",
        "datasets>=2.3.2,<=2.3.2",
        "onnxruntime-gpu>=1.12.1,<=1.12.1;platform_system!='Darwin'",
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
