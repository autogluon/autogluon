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

import sys

version = ag.load_version_file()
version = ag.update_version(version)

submodule = 'tabular'
install_requires = [
    # version ranges added in ag.get_dependency_version_ranges()
    'numpy',  # version range defined in `core/_setup_utils.py`
    'scipy',  # version range defined in `core/_setup_utils.py`
    'pandas',  # version range defined in `core/_setup_utils.py`
    'scikit-learn',  # version range defined in `core/_setup_utils.py`
    'networkx',  # version range defined in `core/_setup_utils.py`
    f'{ag.PACKAGE_NAME}.core=={version}',
    f'{ag.PACKAGE_NAME}.features=={version}',
]

extras_require = {
    'lightgbm': [
        'lightgbm>=3.3,<3.4',
    ],
    'catboost': [
        'catboost>=1.0,<1.2',
    ],
    # FIXME: Debug why xgboost 1.6 has 4x+ slower inference on multiclass datasets compared to 1.4
    #  It is possibly only present on MacOS, haven't tested linux.
    # XGBoost made API breaking changes in 1.6 with custom metric and callback support, so we don't support older versions.
    'xgboost': [
        'xgboost>=1.6,<1.8',
    ],
    'fastai': [
        'torch>=1.9,<1.14',
        'fastai>=2.3.1,<2.8',
    ],
    'ray': [
        f'{ag.PACKAGE_NAME}.core[all]=={version}',
    ],
    'skex': [
        'scikit-learn-intelex>=2021.6,<2021.8',
    ],
    'imodels': [
        'imodels>=1.3.10,<1.4.0',  # 1.3.8/1.3.9 either remove/renamed attribute `complexity_` causing failures. https://github.com/csinva/imodels/issues/147
    ],
    'vowpalwabbit': [
        'vowpalwabbit>=8.10,<8.11'
    ],
    'skl2onnx': [
        'skl2onnx>=1.13.0,<1.14.0',
        # For macOS, there isn't a onnxruntime-gpu package installed with skl2onnx.
        # Therefore, we install onnxruntime explicitly here just for macOS.
        'onnxruntime>=1.13.0,<1.14.0'
    ] if sys.platform == 'darwin' else [
        'skl2onnx>=1.13.0,<1.14.0'
    ]
}

all_requires = []
# TODO: Consider adding 'skex' to 'all'
for extra_package in ['lightgbm', 'catboost', 'xgboost', 'fastai', 'ray']:
    all_requires += extras_require[extra_package]
all_requires = list(set(all_requires))
extras_require['all'] = all_requires


test_requires = []
for test_package in ['imodels', 'vowpalwabbit', 'skl2onnx']:
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
