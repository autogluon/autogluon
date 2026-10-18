from __future__ import annotations

import functools
import logging
import os
import platform
import re
import sys
from datetime import datetime, timezone
from hashlib import md5
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..version import __version__
from .random import get_numpy_seed as _get_numpy_seed

logger = logging.getLogger(__name__)

DEFAULT_BASE_PATH = "AutogluonModels"

#: Env var overriding the directory that auto-generated predictor paths are created under when the
#: user does not specify `path`. Set this to a temporary directory (e.g. in a test fixture) so that
#: predictors created without an explicit path do not leave artifacts in the working directory.
DEFAULT_BASE_PATH_ENV_VAR = "AG_DEFAULT_BASE_PATH"


def get_default_base_path() -> str:
    """Return the base directory used for auto-generated predictor paths.

    Honors the `AG_DEFAULT_BASE_PATH` env var, falling back to `AutogluonModels` in the working
    directory.
    """
    return os.environ.get(DEFAULT_BASE_PATH_ENV_VAR) or DEFAULT_BASE_PATH


def setup_outputdir(
    path: str | Path | None,
    warn_if_exist: bool = True,
    create_dir: bool = True,
    path_suffix: str | None = None,
    default_base_path: str | Path | None = None,
) -> str:
    """Set up the output directory for saving models and results.

    Handles s3 and local paths.

    Parameters
    ----------
    path : str | Path | None
        The base path where models and results will be saved.
        If None, a default path will be created.
    warn_if_exist : bool
        Whether to warn if the specified path already exists and overwriting may occur.
    create_dir : bool
        Whether to create the directory if it does not exist.
    path_suffix : str | None
        A suffix to append to the path. If None, no suffix is added.
    default_base_path : str | Path | None
        A default base path to use if `path` is None.
        If None, defaults to `AutogluonModels` in the current working directory.
        Only used if `path` is None, and thus only used for local paths, not s3 paths.

    Returns
    -------
    path: str
        The absolute local path or s3 path where models and results will be saved.
    """
    if isinstance(path, Path):
        path = str(path)

    if default_base_path is not None:
        if isinstance(default_base_path, Path):
            default_base_path = str(default_base_path)
    else:
        default_base_path = get_default_base_path()

    is_s3_path = False
    if path:
        assert isinstance(path, (str, Path)), (
            f"Only str and pathlib.Path types are supported for path, got {path} of type {type(path)}."
        )

        is_s3_path = str(path).lower().startswith("s3://")

    if path_suffix is None:
        path_suffix = ""
    if path_suffix and path_suffix[-1] == (os.path.sep if not is_s3_path else "/"):
        path_suffix = path_suffix[:-1]

    if path is not None:
        path = f"{path}{path_suffix}"
    else:
        utcnow = datetime.now(timezone.utc)
        timestamp = utcnow.strftime("%Y%m%d_%H%M%S")
        base_name = f"ag-{timestamp}"

        for i in range(1000):
            ag_dir_name = base_name
            if i >= 1:
                ag_dir_name = f"{ag_dir_name}-{i:03d}"
            path = os.path.join(default_base_path, ag_dir_name)
            if path_suffix:
                path = os.path.join(path, path_suffix)
            try:
                if create_dir:
                    os.makedirs(path, exist_ok=False)
                    break
                else:
                    if os.path.isdir(path):
                        raise FileExistsError
                    break
            except FileExistsError:
                pass
        else:
            raise RuntimeError(f"more than 1000 jobs launched in the same second: {path}")
        logger.log(25, f'No path specified. Models will be saved in: "{path}"')
        warn_if_exist = False  # Don't warn about the folder existing since we just created it

    if warn_if_exist and not is_s3_path:
        try:
            if create_dir:
                os.makedirs(path, exist_ok=False)
            elif os.path.isdir(path):
                raise FileExistsError
        except FileExistsError:
            logger.warning(
                f'Warning: path already exists! This predictor may overwrite an existing predictor! path="{path}"'
            )
    if not is_s3_path:
        path = os.path.expanduser(path)  # replace ~ with absolute path if it exists
        path = os.path.abspath(path)
    return path


def get_python_version(include_micro=True) -> str:
    if include_micro:
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    else:
        return f"{sys.version_info.major}.{sys.version_info.minor}"


def get_package_versions(*, strict: bool = False) -> tuple[dict[str, str], list[str]]:
    """
    Return (package_versions, invalid_distributions).

    package_versions:
        Dict of normalized package name -> version for packages that can be read.

    invalid_distributions:
        List of strings describing distributions that could not be read safely
        (e.g., missing/None name metadata, unexpected metadata errors).

    Package names are normalized (lowercase, runs of ``-``, ``_`` and ``.`` collapsed to ``-``), the
    form pip and the wheel file names use, so a name read from a ``.dist-info`` directory and one
    read from a METADATA file compare equal.

    Computed once per process: the installed distributions do not change while a process runs,
    and enumerating them costs up to a second on a large environment, which `TabularPredictor.save`
    would otherwise pay at every fit. The caller gets its own copies of the cached containers.
    """
    package_version_dict, invalid = _get_package_versions_cached(strict=strict)
    return dict(package_version_dict), list(invalid)


_NAME_SEPARATORS = re.compile(r"[-_.]+")
# `{name}-{version}.dist-info`: the wheel spec escapes `-` in both parts to `_`, so the one hyphen
# separates them.
_DIST_INFO_DIR = re.compile(r"^(?P<name>[^-]+)-(?P<version>[^-]+)\.dist-info$")
# `{name}-{version}.egg-info` or `{name}-{version}-py3.11.egg-info`; a bare `{name}.egg-info`
# (an editable install) carries no version and does not match.
_EGG_INFO_DIR = re.compile(r"^(?P<name>[^-]+)-(?P<version>[^-]+)(?:-py\d+(?:\.\d+)?)?\.egg-info$")


def normalize_package_name(name: str) -> str:
    """The normalized form of a distribution name, as pip compares names."""
    return _NAME_SEPARATORS.sub("-", name).lower()


def _name_and_version_from_path(dist) -> tuple[str | None, str | None]:
    """
    Read a distribution's name and version from the name of its metadata directory.

    Installers name the directory `{name}-{version}.dist-info` (or `.egg-info`), so the two values
    are available without opening a file. `(None, None)` for a distribution without a path or with
    a directory name that does not carry a version, in which case the caller reads the metadata.
    """
    base = getattr(dist, "_path", None)
    if base is None:
        return None, None
    directory = Path(base).name
    match = _DIST_INFO_DIR.match(directory) or _EGG_INFO_DIR.match(directory)
    if match is None:
        return None, None
    return match.group("name"), match.group("version")


def _name_and_version_from_metadata_header(dist) -> tuple[str | None, str | None]:
    """
    Read a distribution's Name and Version from the header of its METADATA (or PKG-INFO) file.

    `Distribution.metadata` parses the whole file, long description included, with the email
    parser; the two fields sit in the header, above the first blank line, so reading up to there
    gives the same values at a fraction of the cost. `(None, None)` when the file is not there or
    holds no Name, in which case the caller falls back to `Distribution.metadata`.
    """
    base = getattr(dist, "_path", None)
    if base is None:
        return None, None
    for file_name in ("METADATA", "PKG-INFO"):
        path = Path(base) / file_name
        if not path.is_file():
            continue
        name = version = None
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line in ("\n", "\r\n"):
                    break
                if line.startswith("Name: "):
                    name = line[len("Name: ") :].strip()
                elif line.startswith("Version: "):
                    version = line[len("Version: ") :].strip()
                if name and version:
                    break
        if name:
            return name, version
    return None, None


@functools.lru_cache(maxsize=None)
def _get_package_versions_cached(*, strict: bool) -> tuple[dict[str, str], list[str]]:
    import importlib.metadata

    package_version_dict: dict[str, str] = {}
    invalid: list[str] = []

    for dist in importlib.metadata.distributions():
        try:
            name, version = _name_and_version_from_path(dist)
            if name:
                package_version_dict[normalize_package_name(name)] = version
                continue
            name, version = _name_and_version_from_metadata_header(dist)
            if name:
                package_version_dict[normalize_package_name(name)] = str(version) if version is not None else "unknown"
                continue
            # dist.metadata is typically an email.message.Message-like mapping.
            name = None
            md = getattr(dist, "metadata", None)
            if md is not None:
                # Use .get to avoid KeyError; may still return None.
                try:
                    name = md.get("Name")
                except Exception:
                    # Extremely defensive: some dist objects may have odd metadata implementations.
                    name = None

            # Fall back to Distribution.name if present (py3.8+)
            if not name:
                name = getattr(dist, "name", None)

            if not name:
                invalid.append("Distribution with missing/None name metadata")
                continue

            version = getattr(dist, "version", None)
            if version is None:
                # If version is missing, still record it as unknown rather than crash.
                version = "unknown"

            package_version_dict[normalize_package_name(str(name))] = str(version)
        except Exception as e:
            invalid.append(f"{type(e).__name__}: {e}")
            if strict:
                raise

    return package_version_dict, invalid


def get_autogluon_metadata() -> dict[str, Any]:
    packages, packages_invalid = get_package_versions()
    metadata = dict(
        system=platform.system(),
        version=f"{__version__}",
        py_version=get_python_version(include_micro=False),
        py_version_micro=get_python_version(include_micro=True),
        packages=packages,
        packages_invalid=packages_invalid,
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
        # Metadata written before names were normalized keeps the raw `Name:` casing and separators,
        # so names are matched in normalized form and reported as each side wrote them.
        og_pac = {normalize_package_name(k): (k, v) for k, v in og["packages"].items()}
        cu_pac = {normalize_package_name(k): (k, v) for k, v in cu["packages"].items()}
        for key, (name, version) in og_pac.items():
            if key not in cu_pac:
                logs.append((30, f"WARNING: Missing package '{name}=={version}'"))
            elif version != cu_pac[key][1]:
                logs.append(
                    (30, f"WARNING: Package version diff '{name}'\t(original={version}, current={cu_pac[key][1]})")
                )
        for key, (name, version) in cu_pac.items():
            if key not in og_pac:
                logs.append((30, f"INFO: New package '{name}=={version}'"))

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
            "with the exact AutoGluon version it was created with. AutoGluon does not support backwards compatibility."
        )
        logger.warning(f"\tPredictor Version: {version_saved}")
        logger.warning(f"\tCurrent Version:   {version_current}")
        logger.warning("############################## WARNING ##############################")
        logger.warning("")

        if require_version_match:
            raise AssertionError(
                f"Predictor was created on version {version_saved} but is being loaded with version {version_current}. "
                f"Please ensure the versions match to avoid instability. While it is NOT recommended, "
                f"this error can be bypassed by specifying `require_version_match=False`. "
                f"Exceptions encountered after setting `require_version_match=False` may be very cryptic, "
                f"and in most cases mean that the predictor is fully incompatible with the installed version."
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
    np.random.seed(_get_numpy_seed(seed))
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
