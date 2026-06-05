#!/usr/bin/env python
# Option B thin setup.py: see common/setup.py. Supplies the dynamic version + computed deps
# (caps from _setup_utils.DEPENDENT_PACKAGES + exact `==<version>` sibling pins) and writes version.py.
# Note: package_data (configs/*.yaml) lives in pyproject.toml [tool.setuptools.package-data].
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

submodule = "multimodal"
version = ag.update_version(ag.load_version_file())

install_requires = [
    # version ranges added in ag.get_dependency_version_ranges()
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "Pillow",
    "tqdm",
    "boto3",
    "torch",
    "lightning",
    "transformers[sentencepiece]",
    "accelerate",
    "fsspec[http]<=2025.3",  # pin version to avoid conflicts with `datasets`
    "requests>=2.30,<3",
    "jsonschema>=4.18,<4.24",
    "seqeval>=1.2.2,<1.3.0",
    "evaluate>=0.4.0,<0.5.0",
    "timm>=0.9.5,<1.0.7",
    "torchvision>=0.21.0,<0.25.0",
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

extras_require = {
    "tests": [
        "ruff",
        "datasets>=2.16.0,<3.6.0",
        "tensorrt>=8.6.0,<10.9.1;platform_system=='Linux' and python_version<'3.11'",
        # Sync ONNX requirements with tabular/setup.py
        "onnx>=1.13.0,!=1.16.2,<1.21.0;platform_system=='Windows'",  # exclude 1.16.2 for issue https://github.com/onnx/onnx/issues/6267
        "onnx>=1.13.0,<1.21.0;platform_system!='Windows'",
        # For macOS, there isn't a onnxruntime-gpu package installed with skl2onnx.
        # Therefore, we install onnxruntime explicitly here just for macOS.
        "onnxruntime>=1.17.0,<1.24.0",
        "onnxruntime-gpu>=1.17.0,<1.24.0; platform_system != 'Darwin' and platform_machine != 'aarch64'",
    ],
}

install_requires = ag.get_dependency_version_ranges(install_requires)

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
