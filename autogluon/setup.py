#!/usr/bin/env python
###########################
# This code block is a HACK (!), but is necessary to avoid code duplication. Do NOT alter these lines.
import os
from setuptools import setup
import importlib.util
filepath = os.path.abspath(os.path.dirname(__file__))
filepath_import = os.path.join(filepath, '..', 'core', 'src', 'autogluon', 'core', '_setup_utils.py')
if not os.path.exists(filepath_import):
    filepath_import = os.path.join(filepath, "_setup_utils.py")

spec = importlib.util.spec_from_file_location("ag_min_dependencies", filepath_import)
ag = importlib.util.module_from_spec(spec)
# Identical to `from autogluon.core import _setup_utils as ag`, but works without `autogluon.core` being installed.
spec.loader.exec_module(ag)
###########################

version = ag.load_version_file()
version = ag.update_version(version)

submodule = None  # None since this module is special (it isn't a real submodule)
install_requires = [
    f'autogluon.core[all]=={version}',
    f'autogluon.features=={version}',
    f'autogluon.tabular[all]=={version}',
    f'autogluon.multimodal=={version}',
    f'autogluon.timeseries[all]=={version}',
] if not ag.LITE_MODE else [
    f'{ag.PACKAGE_NAME}.core=={version}',
    f'{ag.PACKAGE_NAME}.features=={version}',
    f'{ag.PACKAGE_NAME}.tabular=={version}',
]

install_requires = ag.get_dependency_version_ranges(install_requires)

extras_require = {
    "tabarena": [f'autogluon.tabular[tabarena]=={version}']
}
if __name__ == '__main__':
    ag.create_version_file(version=version, submodule=submodule)
    setup_args = ag.default_setup_args(version=version, submodule=submodule)
    setup(
        install_requires=install_requires,
        extras_require=extras_require,
        **setup_args,
    )
