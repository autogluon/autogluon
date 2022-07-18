#!/usr/bin/env python
import warnings

from packaging.version import parse as vparse

###########################
# This code block is a HACK (!), but is necessary to avoid code duplication. Do NOT alter these lines.
import os
from setuptools import setup
import importlib.util

filepath = os.path.abspath(os.path.dirname(__file__))
filepath_import = os.path.join(
    filepath, "..", "core", "src", "autogluon", "core", "_setup_utils.py"
)
spec = importlib.util.spec_from_file_location("ag_min_dependencies", filepath_import)
ag = importlib.util.module_from_spec(spec)
# Identical to `from autogluon.core import _setup_utils as ag`, but works without `autogluon.core` being installed.
spec.loader.exec_module(ag)
###########################

version = ag.load_version_file()
version = ag.update_version(version)

submodule = "timeseries"
install_requires = [
    # version ranges added in ag.get_dependency_version_ranges()
    "numpy",
    "scipy",
    "pandas",
    "psutil>=5.7.3,<5.9",
    "gluonts>=0.8.0,<0.10.0",
    f"autogluon.core=={version}",
    f"autogluon.common=={version}",
]

try:
    from mxnet import __version__ as mxnet_version

    assert vparse("2.0") > vparse(mxnet_version) >= vparse("1.9")
except (ImportError, AssertionError):
    warnings.warn(
        "autogluon.forecasting depends on Apache MXNet v1.9 or greater (below v2.0). "
        "Please install a suitable version of MXNet in order to use autogluon.forecasting using "
        "`pip install mxnet==1.9` or a matching MXNet package for your CUDA driver if you are using "
        "a GPU. See the MXNet documentation for more info."
    )

extras_require = {
    "tests": ["pytest", "flake8~=4.0", "flaky~=3.7", "pytest-timeout~=2.1"],
    "sktime": ["sktime~=0.11.4", "pmdarima~=1.8.2", "tbats~=1.1"],
}

all_requires = []
for extra_package in ["sktime"]:
    all_requires += extras_require[extra_package]
extras_require['all'] = list(set(all_requires))

install_requires = ag.get_dependency_version_ranges(install_requires)

if __name__ == "__main__":
    ag.create_version_file(version=version, submodule=submodule)
    setup_args = ag.default_setup_args(version=version, submodule=submodule)
    setup(
        install_requires=install_requires,
        extras_require=extras_require,
        **setup_args,
    )
