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
    'networkx',
    'pandas',
    'tqdm',

    'requests',
    'matplotlib',
    'boto3',

    f'autogluon.common=={version}',
] if not ag.LITE_MODE else [
    # version ranges added in ag.get_dependency_version_ranges()
    'numpy',
    'scipy',
    'scikit-learn',
    'pandas',
    'tqdm',
    'matplotlib',

    f'{ag.PACKAGE_NAME}.common=={version}',
]

extras_require = {
    'ray': [
        "ray>=2.2,<2.3",
    ],
    'raytune': [
        'ray[tune]>=2.2,<2.3',
        'hyperopt>=0.2.7,<0.2.8',
        # 'GPy>=1.10.0,<1.11.0'  # TODO: Enable this once PBT/PB2 are supported by ray lightning
    ]
}

tests_require = [
    'pytest',
    'types-requests',
    'types-setuptools',
    'pytest-mypy',
    # TODO(Re-enable ray_lightning once it released 0.3.0) 'ray_lightning>=0.2.0,<0.3.0'
]

all_requires = []

for extra_package in ['ray', 'raytune']:
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
