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
install_requires = [
    # version ranges added in ag.get_dependency_version_ranges()
    'numpy',
    'scipy',
    'pandas',
    'scikit-learn',
    'psutil',
    'networkx>=2.3,<3.0',
    f'autogluon.core=={version}',
    f'autogluon.features=={version}',
]

extras_require = {
    'lightgbm': [
        'lightgbm>=3.3,<3.4',
    ],
    'catboost': [
        'catboost>=1.0,<1.1',
    ],
    # FIXME: Debug why xgboost 1.6 has 4x+ slower inference on multiclass datasets compared to 1.4
    #  It is possibly only present on MacOS, haven't tested linux.
    'xgboost': [
        'xgboost>=1.4,<1.5',
    ],
    'fastai': [
        'torch>=1.0,<1.13',
        'fastai>=2.3.1,<2.8',
    ],
    'skex': [
        'scikit-learn-intelex>=2021.5,<2021.6',
    ],
    'imodels': [
        'imodels>=1.3.0',
    ],
    'vowpalwabbit': [
        'vowpalwabbit>=8.10,<8.11'
    ]
}

all_requires = []
# TODO: Consider adding 'skex' to 'all'
for extra_package in ['lightgbm', 'catboost', 'xgboost', 'fastai']:
    all_requires += extras_require[extra_package]
all_requires = list(set(all_requires))
extras_require['all'] = all_requires


test_requires = []
for test_package in ['imodels', 'vowpalwabbit']:
    test_requires += extras_require[test_package]
extras_require['tests'] = test_requires
install_requires = ag.get_dependency_version_ranges(install_requires)

if __name__ == '__main__':
    ag.create_version_file(version=version, submodule=submodule)
    setup_args = ag.default_setup_args(version=version, submodule=submodule)
    setup(
        install_requires=install_requires,
        extras_require=extras_require,
        **setup_args,
    )
