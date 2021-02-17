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

submodule = 'tabular'
requirements = [
    # version ranges added in ag.get_dependency_version_ranges()
    'numpy',
    'scipy',
    'pandas',
    'scikit-learn',

    'catboost>=0.23.0,<0.25',
    'xgboost>=1.3.2,<1.4',
    'lightgbm>=3.0,<4.0',
    'psutil>=5.0.0,<=5.7.0',  # TODO: psutil 5.7.1/5.7.2 has non-deterministic error on CI doc build -  ImportError: cannot import name '_psutil_linux' from 'psutil'
    'networkx>=2.3,<3.0',
    'torch>=1.0,<2.0',  # TODO: v0.1 make optional
    'fastai>=1.0,<2.0',  # TODO: v0.1 make optional
    f'autogluon.core=={version}',
    f'autogluon.features=={version}',
]

test_requirements = [
    'pytest',
]

install_requires = requirements + test_requirements
install_requires = ag.get_dependency_version_ranges(install_requires)

if __name__ == '__main__':
    ag.create_version_file(version=version, submodule=submodule)
    setup_args = ag.default_setup_args(version=version, submodule=submodule)
    setup(
        install_requires=install_requires,
        **setup_args,
    )
