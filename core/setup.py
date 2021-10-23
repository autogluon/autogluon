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
version = ag.update_version(version, use_file_if_exists=False, create_file=True)

submodule = 'core'
install_requires = [
    # version ranges added in ag.get_dependency_version_ranges()
    'numpy',
    'scipy',
    'scikit-learn',
    'pandas',
    'tqdm',
    'graphviz',

    'cython',  # TODO: Do we need cython here? Why is cython not version capped / minned?
    'ConfigSpace==0.4.19',
    'tornado>=5.0.1',
    'requests',
    'matplotlib',
    'paramiko>=2.4',
    'dask>=2.6.0',
    'distributed>=2.6.0',
    'boto3',
    'autograd>=1.3',
    'dill>=0.3.3,<1.0',
]

extras_require = {
    'extra_searchers': [
        'scikit-optimize',  # Optional due to only being rarely used and due to breaking install in the past
    ],
}

tests_require = [
    'pytest',
]
for extra_package in ['extra_searchers']:
    tests_require += extras_require[extra_package]
tests_require = list(set(tests_require))
extras_require['tests'] = tests_require

install_requires = ag.get_dependency_version_ranges(install_requires)

if __name__ == '__main__':
    ag.create_version_file(version=version, submodule=submodule)
    setup_args = ag.default_setup_args(version=version, submodule=submodule)
    setup(
        install_requires=install_requires,
        extras_require=extras_require,
        entry_points={
            'console_scripts': [
                'agremote = autogluon.core.scheduler.remote.cli:main',
            ]
        },
        **setup_args,
    )
