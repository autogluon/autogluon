import logging
import os
import platform
import sys
from datetime import datetime
from hashlib import md5
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..version import __version__

try:
    from ..version import __lite__
except ImportError:
    __lite__ = False

logger = logging.getLogger(__name__)

LITE_MODE: bool = __lite__ is not None and __lite__


def setup_outputdir(path, warn_if_exist=True, create_dir=True, path_suffix=None):
    if path:
        assert isinstance(path, (str, Path)), f"Only str and pathlib.Path types are supported for path, got {path} of type {type(path)}."
    if path_suffix is None:
        path_suffix = ""
    if path_suffix and path_suffix[-1] == os.path.sep:
        path_suffix = path_suffix[:-1]
    if path is not None:
        path = f"{path}{path_suffix}"
    if path is None:
        utcnow = datetime.utcnow()
        timestamp = utcnow.strftime("%Y%m%d_%H%M%S")
        path = os.path.join("AutogluonModels", f"ag-{timestamp}{path_suffix}")
        for i in range(1, 1000):
            try:
                if create_dir:
                    os.makedirs(path, exist_ok=False)
                    break
                else:
                    if os.path.isdir(path):
                        raise FileExistsError
                    break
            except FileExistsError:
                path = os.path.join("AutogluonModels", f"ag-{timestamp}-{i:03d}{path_suffix}")
        else:
            raise RuntimeError("more than 1000 jobs launched in the same second")
        logger.log(25, f'No path specified. Models will be saved in: "{path}"')
    elif warn_if_exist:
        try:
            if create_dir:
                os.makedirs(path, exist_ok=False)
            elif os.path.isdir(path):
                raise FileExistsError
        except FileExistsError:
            logger.warning(f'Warning: path already exists! This predictor may overwrite an existing predictor! path="{path}"')
    path = os.path.expanduser(path)  # replace ~ with absolute path if it exists
    return path


def get_python_version(include_micro=True) -> str:
    if include_micro:
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    else:
        return f"{sys.version_info.major}.{sys.version_info.minor}"


def get_package_versions() -> Dict[str, str]:
    """Gets a dictionary of package name -> package version for every package installed in the environment"""
    import pkg_resources

    package_dict = pkg_resources.working_set.by_key  # type: ignore
    package_version_dict = {key: val.version for key, val in package_dict.items()}
    return package_version_dict


def get_autogluon_metadata() -> Dict[str, Any]:
    metadata = dict(
        system=platform.system(),
        version=f"{__version__}",
        lite=__lite__,
        py_version=get_python_version(include_micro=False),
        py_version_micro=get_python_version(include_micro=True),
        packages=get_package_versions(),
    )
    return metadata


def compare_autogluon_metadata(*, original: dict, current: dict, check_packages=True) -> list:
    logs = []
    og = original
    cu = current
    if og["version"] != cu["version"]:
        logs.append((30, f"WARNING: AutoGluon version mismatch (original={og['version']}, current={cu['version']})"))
    if og["py_version"] != cu["py_version"]:
        logs.append(
            (
                30,
                f"WARNING: AutoGluon Python version mismatch (original={og['py_version']}, current={cu['py_version']})",
            )
        )
    elif og["py_version_micro"] != cu["py_version_micro"]:
        logs.append(
            (
                30,
                f"INFO: AutoGluon Python micro version mismatch (original={og['py_version_micro']}, current={cu['py_version_micro']})",
            )
        )
    if og["system"] != cu["system"]:
        logs.append((30, f"WARNING: System mismatch (original={og['system']}, current={cu['system']})"))
    if check_packages:
        og_pac = og["packages"]
        cu_pac = cu["packages"]
        for k in og_pac.keys():
            if k not in cu_pac:
                logs.append((30, f"WARNING: Missing package '{k}=={og_pac[k]}'"))
            elif og_pac[k] != cu_pac[k]:
                logs.append((30, f"WARNING: Package version diff '{k}'\t(original={og_pac[k]}, current={cu_pac[k]})"))
        for k in cu_pac.keys():
            if k not in og_pac:
                logs.append((30, f"INFO: New package '{k}=={cu_pac[k]}'"))

    if len(logs) > 0:
        logger.log(30, f"Found {len(logs)} mismatches between original and current metadata:")
    for log in logs:
        logger.log(log[0], f"\t{log[1]}")

    return logs


def bytes_to_mega_bytes(memory_amount: int) -> int:
    """Utility to convert a number of bytes (int) into a number of mega bytes (int)"""
    return memory_amount >> 20


def check_saved_predictor_version(
    version_current: str,
    version_saved: str,
    require_version_match: bool = True,
    logger: Optional[logging.Logger] = None,
) -> None:
    if logger is None:
        logger = logging.getLogger(__name__)

    if version_saved != version_current:
        logger.warning("")
        logger.warning("############################## WARNING ##############################")
        logger.warning(
            "WARNING: AutoGluon version differs from the version used to create the predictor! "
            "This may lead to instability and it is highly recommended the predictor be loaded "
            "with the exact AutoGluon version it was created with."
        )
        logger.warning(f"\tPredictor Version: {version_saved}")
        logger.warning(f"\tCurrent Version:   {version_current}")
        logger.warning("############################## WARNING ##############################")
        logger.warning("")

        if require_version_match:
            raise AssertionError(
                f"Predictor was created on version {version_saved} but is being loaded with version {version_current}. "
                f"Please ensure the versions match to avoid instability. While it is NOT recommended, "
                f"this error can be bypassed by specifying `require_version_match=False`."
            )


def hash_pandas_df(df: Optional[pd.DataFrame]) -> str:
    """Compute a hash string for a pandas DataFrame."""
    if df is not None:
        # Convert in case TimeSeriesDataFrame object is passed
        df = pd.DataFrame(df, copy=True)
        df.reset_index(inplace=True)
        df.sort_index(inplace=True, axis=1)
        hashable_object = pd.util.hash_pandas_object(df).values
    else:
        hashable_object = "0".encode("utf-8")
    return md5(hashable_object).hexdigest()


def seed_everything(seed: int) -> None:
    """Set random seeds for numpy and PyTorch."""
    logger.debug(f"Setting random seed to {seed}")
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
