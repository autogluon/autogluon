import autogluon
import importlib
import pkgutil
import pkg_resources
import platform
import sys

from datetime import datetime

from .nvutil import cudaInit, cudaSystemGetNVMLVersion
from .utils import (
    get_available_disk_size,
    get_cpu_count,
    get_gpu_count_all,
    get_gpu_free_memory,
    get_memory_size
)


# We don't include test dependency here
autogluon_extras_dict = {
    'autogluon.core': ('all',),
    'autogluon.common': (),
    'autogluon.features': (),
    'autogluon.forecasting': (),
    'autogluon.tabular': ('all',),
    'autogluon.text': (),
    'autogluon.vision': (),
}

# This is needed because some module are different in its import name and pip install name
import_name_dict = {
    'autogluon-contrib-nlp': 'autogluon_contrib_nlp',
    'pillow': 'PIL',
    'pytorch-lightning': 'pytorch_lightning',
    'scikit-image': 'skimage',
    'scikit-learn': 'sklearn',
    'smart-open': 'smart_open',
    'timm-clean': 'timm',
}


def _get_autogluon_versions():
    """Retrieve version of all autogluon subpackages and its dependencies"""
    versions = dict()
    for pkg in list(pkgutil.iter_modules(autogluon.__path__, autogluon.__name__ + '.')):
        if pkg.name == 'autogluon.version':  # autogluon.version will be recognized as a submodule by pkgutil. We don't need it
            continue
        try:
            module = importlib.import_module(pkg.name)
            versions[pkg.name] = module.__version__
            versions.update(_get_dependency_versions(pkg.name, autogluon_extras_dict.get(pkg.name, ())))
        except ImportError:
            versions[pkg.name] = None
    return versions


def _get_dependency_versions(package, extras=()):
    """Retrieve direct dependency of the given package

    Args:
        package (str): name of the package
        extras (tuple, optional): extras in package dependency. Defaults to ().
    """
    package = pkg_resources.working_set.by_key[package]
    dependencies = [str(r.key) for r in package.requires(extras=extras)]
    versions = dict()
    for dependency in dependencies:
        dependency = import_name_dict.get(dependency, dependency)
        try:
            module = importlib.import_module(dependency)
            version = getattr(module, "__version__", None)
            if version is None:
                version = getattr(module, "__VERSION__", None)
            versions[dependency] = version
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
        'date': datetime.date(datetime.now()),
        'time': datetime.time(datetime.now()),
        'python': '.'.join(str(i) for i in sys.version_info),
        'OS': uname.system,
        'OS-release': uname.release,
        'Version': uname.version,
        'machine': uname.machine,
        'processor': uname.processor,
        'num_cores': get_cpu_count(),
        'cpu_ram_mb': get_memory_size(),
        'cuda version': cuda_version,
        'num_gpus': get_gpu_count_all(),
        'gpu_ram_mb': get_gpu_free_memory(),
        'avail_disk_size_mb': get_available_disk_size(),
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

    maxlen = max(len(x) for x in versions)
    print("\nINSTALLED VERSIONS")
    print("------------------")
    for k, v in sys_info.items():
        print(f"{k:<{maxlen}}: {v}")
    print("")
    for k in sorted_keys:
        print(f"{k:<{maxlen}}: {versions[k]}")
