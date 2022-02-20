#!/usr/bin/env python
###########################
# This code block is a HACK (!), but is necessary to avoid code duplication. Do NOT alter these lines.
import os
from setuptools import setup
import importlib.util
filepath = os.path.abspath(os.path.dirname(__file__))
filepath_import = os.path.join(filepath, '..', 'core', 'src', 'autogluon', 'core', '_setup_utils.py')
spec = importlib.util.spec_from_file_location("ag_min_dependencies", filepath_import)
ag = importlib.util.module_from_spec(spec)
# Identical to `from autogluon.core import _setup_utils as ag`, but works without `autogluon.core` being installed.
spec.loader.exec_module(ag)
###########################

version = ag.load_version_file()
version = ag.update_version(version)

submodule = 'text'
install_requires = [
    # version ranges added in ag.get_dependency_version_ranges()
    'numpy',
    'scipy',
    'pandas',
    'scikit-learn',
    'Pillow',
    'tqdm',
    'fairscale<0.4.6',
    'scikit-image<0.19.3',
    'boto3<1.21.4',
    'smart_open<5.2.2',
    'pytorch_lightning<1.5.11',
    'torchmetrics<0.7.3',
    'transformers<4.16.3',
    'nptyping<1.4.5',
    'omegaconf<2.1.2',
    'timm<0.5.5',
    'sentencepiece<0.1.97',
    f'autogluon.core=={version}',
    'autogluon-contrib-nlp==0.0.1b20220208',
]

install_requires = ag.get_dependency_version_ranges(install_requires)

if __name__ == '__main__':
    ag.create_version_file(version=version, submodule=submodule)
    setup_args = ag.default_setup_args(version=version, submodule=submodule)
    setup_args["package_data"]["autogluon.text.automm"] = [
        'configs/data/*.yaml',
        'configs/model/*.yaml',
        'configs/optimization/*.yaml',
        'configs/environment/*.yaml',
    ]
    setup(
        install_requires=install_requires,
        **setup_args,
    )
