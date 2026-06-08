#!/usr/bin/env python
# Option B thin setup.py for the meta package (submodule=None): see common/setup.py. Supplies the
# dynamic version + computed deps (exact `==<version>` sibling pins) and writes version.py.
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

submodule = None  # meta package
version = ag.update_version(ag.load_version_file())

install_requires = [
    f"autogluon.core[all]=={version}",
    f"autogluon.features=={version}",
    f"autogluon.tabular[all]=={version}",
    f"autogluon.multimodal=={version}",
    f"autogluon.timeseries[all]=={version}",
]

extras_require = {
    "tabarena": [f"autogluon.tabular[tabarena]=={version}"],
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
