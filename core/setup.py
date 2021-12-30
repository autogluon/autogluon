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

submodule = 'core'
install_requires = [
    # version ranges added in ag.get_dependency_version_ranges()
    'numpy',
    'scipy',
    'scikit-learn',
    'pandas',
    'tqdm',

    'requests',
    'matplotlib',
    # dask and distributed==2021.12.0 will cause ray(1.7.0 - 1.9.0) to fail
    # error:
    # https://ci.gluon.ai/blue/organizations/jenkins/autogluon/detail/master/702/pipeline/16
    'dask>=2021.09.1,<=2021.11.2',
    'distributed>=2021.09.1, <=2021.11.2',
    'boto3',

    f'autogluon.common=={version}',
]

extras_require = {
    'ray': [
        'ray>=1.7,<1.8',
    ],
}

tests_require = [
    'pytest',
]

all_requires = []

for extra_package in ['ray']:
    all_requires += extras_require[extra_package]
tests_require = list(set(tests_require))
all_requires = list(set(all_requires))
extras_require['tests'] = tests_require
extras_require['all'] = all_requires

install_requires = ag.get_dependency_version_ranges(install_requires)

if __name__ == '__main__':
    ag.create_version_file(version=version, submodule=submodule)
    setup_args = ag.default_setup_args(version=version, submodule=submodule)
    setup(
        install_requires=install_requires,
        extras_require=extras_require,
        **setup_args,
    )
