from pathlib import Path
from datetime import datetime

import logging
import os
import platform
import sys
from typing import Dict, Any

logger = logging.getLogger(__name__)


def setup_outputdir(path, warn_if_exist=True, create_dir=True, path_suffix=None):
    if path:
        assert isinstance(path, (str, Path)), \
            f"Only str and pathlib.Path types are supported for path, got {path} of type {type(path)}."
    if path_suffix is None:
        path_suffix = ''
    if path_suffix and path_suffix[-1] == os.path.sep:
        path_suffix = path_suffix[:-1]
    if path is not None:
        path = f'{path}{path_suffix}'
    if path is None:
        utcnow = datetime.utcnow()
        timestamp = utcnow.strftime("%Y%m%d_%H%M%S")
        path = f"AutogluonModels/ag-{timestamp}{path_suffix}{os.path.sep}"
        for i in range(1, 1000):
            try:
                if create_dir:
                    os.makedirs(path, exist_ok=False)
                    break
                else:
                    if os.path.isdir(path):
                        raise FileExistsError
                    break
            except FileExistsError as e:
                path = f"AutogluonModels/ag-{timestamp}-{i:03d}{path_suffix}{os.path.sep}"
        else:
            raise RuntimeError("more than 1000 jobs launched in the same second")
        logger.log(25, f'No path specified. Models will be saved in: "{path}"')
    elif warn_if_exist:
        try:
            if create_dir:
                os.makedirs(path, exist_ok=False)
            elif os.path.isdir(path):
                raise FileExistsError
        except FileExistsError as e:
            logger.warning(f'Warning: path already exists! This predictor may overwrite an existing predictor! path="{path}"')
    path = os.path.expanduser(path)  # replace ~ with absolute path if it exists
    if path[-1] != os.path.sep:
        path = path + os.path.sep
    return path


def get_python_version(include_micro=True) -> str:
    if include_micro:
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    else:
        return f"{sys.version_info.major}.{sys.version_info.minor}"


def get_package_versions() -> Dict[str, str]:
    """Gets a dictionary of package name -> package version for every package installed in the environment"""
    import pkg_resources
    package_dict = pkg_resources.working_set.by_key
    package_version_dict = {key: val.version for key, val in package_dict.items()}
    return package_version_dict


def get_autogluon_metadata() -> Dict[str, Any]:
    from ..version import __version__, __lite__
    metadata = dict(
        system=platform.system(),
        version=f'{__version__}',
        lite=__lite__,
        py_version=get_python_version(include_micro=False),
        py_version_micro=get_python_version(include_micro=True),
        packages=get_package_versions(),
    )
    return metadata


def disable_if_lite_mode(ret=None):
    def inner(func):
        def do_nothing(*args, **kwargs):
            if callable(ret):
                return ret(*args, **kwargs)
            return ret
        metadata = get_autogluon_metadata()
        if metadata['lite']:
            return do_nothing
        return func
    return inner


def compare_autogluon_metadata(*, original: dict, current: dict, check_packages=True) -> list:
    logs = []
    og = original
    cu = current
    if og['version'] != cu['version']:
        logs.append((30, f"WARNING: AutoGluon version mismatch (original={og['version']}, current={cu['version']})"))
    if og['py_version'] != cu['py_version']:
        logs.append((30, f"WARNING: AutoGluon Python version mismatch (original={og['py_version']}, current={cu['py_version']})"))
    elif og['py_version_micro'] != cu['py_version_micro']:
        logs.append((30, f"INFO: AutoGluon Python micro version mismatch (original={og['py_version_micro']}, current={cu['py_version_micro']})"))
    if og['system'] != cu['system']:
        logs.append((30, f"WARNING: System mismatch (original={og['system']}, current={cu['system']})"))
    if check_packages:
        og_pac = og['packages']
        cu_pac = cu['packages']
        for k in og_pac.keys():
            if k not in cu_pac:
                logs.append((30, f"WARNING: Missing package '{k}=={og_pac[k]}'"))
            elif og_pac[k] != cu_pac[k]:
                logs.append((30, f"WARNING: Package version diff '{k}'\t(original={og_pac[k]}, current={cu_pac[k]})"))
        for k in cu_pac.keys():
            if k not in og_pac:
                logs.append((30, f"INFO: New package '{k}=={cu_pac[k]}'"))

    if len(logs) > 0:
        logger.log(30, f'Found {len(logs)} mismatches between original and current metadata:')
    for log in logs:
        logger.log(log[0], f"\t{log[1]}")

    return logs
