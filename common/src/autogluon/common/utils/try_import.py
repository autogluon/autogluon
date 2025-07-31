import logging
import os
import platform
import sys
from types import ModuleType

from ..version import __version__

__all__ = [
    "try_import_mxboard",
    "try_import_catboost",
    "try_import_lightgbm",
    "try_import_xgboost",
    "try_import_faiss",
    "try_import_fastai",
    "try_import_torch",
    "try_import_autogluon_multimodal",
    "try_import_rapids_cuml",
    "try_import_imodels",
    "try_import_fasttext",
]

logger = logging.getLogger(__name__)


def try_import_mxboard():
    try:
        import mxboard
    except ImportError:
        raise ImportError("Unable to import dependency mxboard. A quick tip is to install via `pip install mxboard`. ")


def try_import_ray() -> ModuleType:
    RAY_MAX_VERSION = "2.45.0"  # sync with core/setup.py
    ray_max_version_os_map = dict(
        Darwin=RAY_MAX_VERSION,
        Windows=RAY_MAX_VERSION,
        Linux=RAY_MAX_VERSION,
    )
    ray_min_version = "2.10.0"
    current_os = platform.system()
    ray_max_version = ray_max_version_os_map.get(current_os, RAY_MAX_VERSION)
    strict_ray_version = os.environ.get("AG_LOOSE_RAY_VERSION", "False") != "True"
    try:
        import ray
        from packaging import version

        if (
            version.parse(ray.__version__) < version.parse(ray_min_version)
            or version.parse(ray.__version__) >= version.parse(ray_max_version)
        ) and strict_ray_version:
            msg = (
                f"ray=={ray.__version__} detected. "
                f"{ray_min_version} <= ray < {ray_max_version} is required. You can use pip to install certain version of ray "
                f'`pip install "ray>={ray_min_version},<{ray_max_version}"`'
            )
            raise ValueError(msg)
        return ray
    except ImportError:
        raise ImportError(
            "ray is required to train folds in parallel for TabularPredictor or HPO for MultiModalPredictor. "
            f'A quick tip is to install via `pip install "ray>={ray_min_version},<{ray_max_version}"`'
        )


def try_import_catboost():
    try:
        import catboost
        from packaging import version

        catboost_version = version.parse(catboost.__version__)
        min_version = "1.2"
        assert catboost_version >= version.parse(min_version), (
            f'Currently, we support "catboost>={min_version}". Installed version: "catboost=={catboost.__version__}".'
        )
    except ImportError as e:
        raise ImportError(
            "`import catboost` failed. "
            f"A quick tip is to install via `pip install autogluon.tabular[catboost]=={__version__}`."
        ) from e
    except ValueError as e:
        raise ImportError(
            "Import catboost failed. Numpy version may be outdated, "
            "Please ensure numpy version >=1.17.0. If it is not, please try 'pip uninstall numpy -y; pip install numpy>=1.17.0' "
            "Detailed info: {}".format(str(e))
        ) from e


def try_import_lightgbm():
    try:
        import lightgbm
    except ImportError as e:
        raise ImportError(
            "`import lightgbm` failed. "
            f"A quick tip is to install via `pip install autogluon.tabular[lightgbm]=={__version__}`."
        )
    except OSError as e:
        raise ImportError(
            "`import lightgbm` failed. If you are using Mac OSX, "
            "Please try 'brew install libomp'. Detailed info: {}".format(str(e))
        )


def try_import_xgboost():
    try:
        import xgboost
        from packaging import version

        xgboost_version = version.parse(xgboost.__version__)
        min_version = "1.6"
        assert xgboost_version >= version.parse(min_version), (
            f'Currently, we only support "xgboost>={min_version}". Installed version: "xgboost=={xgboost.__version__}".'
        )
    except ImportError:
        raise ImportError(
            "`import xgboost` failed. "
            f"A quick tip is to install via `pip install autogluon.tabular[xgboost]=={__version__}`."
        )


def try_import_faiss():
    try:
        import faiss
    except ImportError:
        raise ImportError("Unable to import dependency faiss. A quick tip is to install via `pip install faiss-cpu`. ")


def try_import_fastai():
    try:
        import fastai
        from packaging import version

        fastai_version = version.parse(fastai.__version__)
        assert version.parse("2.0.0") <= fastai_version, "Currently, we only support fastai>=2.0.0"

        # fastai is doing library setup during star imports. These are required for correct library functioning.
        # Local star imports are not possible in-place, so separate helper packages is created
        import autogluon.tabular.models.fastainn.imports_helper

    except ModuleNotFoundError as e:
        raise ImportError(
            f"Import fastai failed. A quick tip is to install via `pip install autogluon.tabular[fastai]=={__version__}`. "
        )


def try_import_torch():
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "Unable to import dependency torch\n"
            "A quick tip is to install via `pip install torch`.\n"
            "The minimum torch version is currently 2.6."  # sync with core/_setup_utils.py
        )


def try_import_autogluon_multimodal():
    try:
        import autogluon.multimodal
    except ImportError:
        raise ImportError(
            "`import autogluon.multimodal` failed.\n"
            f"A quick tip is to install via `pip install autogluon.multimodal=={__version__}`.\n"
        )


def try_import_rapids_cuml():
    try:
        import cuml
    except ImportError:
        raise ImportError(
            "`import cuml` failed.\n"
            "Ensure that you have a GPU and CUDA installation, and then install RAPIDS.\n"
            "You will likely need to create a fresh conda environment based off of a RAPIDS install, and then install AutoGluon on it.\n"
            "RAPIDS is highly experimental within AutoGluon, and we recommend to only use RAPIDS if you are an advanced user / developer.\n"
            "Please refer to RAPIDS install instructions for more information: https://rapids.ai/start.html#get-rapids"
        )


def try_import_imodels():
    try:
        import imodels
    except ImportError:
        raise ImportError("Unable to import dependency imodels. A quick tip is to install via `pip install imodels`. ")


def try_import_fasttext():
    try:
        import fasttext

        _ = fasttext.__file__
    except Exception:
        raise ImportError('Import fasttext failed. Please run "pip install fasttext"')
