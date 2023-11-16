import pkgutil
import platform
import re
import sys
from datetime import datetime
from importlib.metadata import distribution, version

import autogluon
from autogluon.common.utils.nvutil import cudaInit, cudaSystemGetNVMLVersion
from autogluon.common.utils.resource_utils import ResourceManager


def _get_autogluon_versions():
    """Retrieve version of all autogluon subpackages and its dependencies"""
    versions = dict()
    for pkg in list(pkgutil.iter_modules(autogluon.__path__, autogluon.__name__ + ".")):
        # The following packages will be recognized as a submodule by pkgutil -exclude them.
        if pkg.name in ["autogluon.version", "autogluon.setup", "autogluon._internal_"]:
            continue
        try:
            versions[pkg.name] = version(pkg.name)
            versions.update(_get_dependency_versions(pkg.name))
        except ImportError:
            versions[pkg.name] = None
    return versions


def _get_dependency_versions(package):
    """Retrieve direct dependency of the given package

    Args:
        package (str): name of the package
    """
    # Get all requires for the package
    dependencies = distribution(package).requires
    # Filter-out test dependencies
    dependencies = [req for req in dependencies if not bool(re.search("extra.*test", req))]
    # keep only package name
    dependencies = [re.findall("[a-zA-Z0-9_\\-]+", req)[0].strip() for req in dependencies]
    versions = dict()
    for dependency in dependencies:
        try:
            versions[dependency] = version(dependency)
        except ImportError:
            versions[dependency] = None
    return versions


def _get_sys_info():
    """Retrieve system information"""
    uname = platform.uname()
    cuda_version = None
    if cudaInit():
        try:
            cuda_version = cudaSystemGetNVMLVersion()
        except:
            cuda_version = None

    return {
        "date": datetime.date(datetime.now()),
        "time": datetime.time(datetime.now()),
        "python": ".".join(str(i) for i in sys.version_info),
        "OS": uname.system,
        "OS-release": uname.release,
        "Version": uname.version,
        "machine": uname.machine,
        "processor": uname.processor,
        "num_cores": ResourceManager.get_cpu_count(),
        "cpu_ram_mb": ResourceManager.get_memory_size("MB"),
        "cuda version": cuda_version,
        "num_gpus": ResourceManager.get_gpu_count(),
        "gpu_ram_mb": ResourceManager.get_gpu_free_memory(),
        "avail_disk_size_mb": ResourceManager.get_available_disk_size(),
    }


def show_versions():
    """
    Provide useful information, important for bug reports.
    It comprises info about hosting operation system, autogluon subpackage versions,
    and versions of other installed relative packages.
    """
    sys_info = _get_sys_info()
    versions = _get_autogluon_versions()
    sorted_keys = sorted(versions.keys(), key=lambda x: x.lower())

    maxlen = 0 if len(versions) == 0 else max(len(x) for x in versions)
    print("\nINSTALLED VERSIONS")
    print("------------------")
    for k, v in sys_info.items():
        print(f"{k:<{maxlen}}: {v}")
    print("")
    for k in sorted_keys:
        print(f"{k:<{maxlen}}: {versions[k]}")
