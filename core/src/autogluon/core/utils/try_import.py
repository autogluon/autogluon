import logging
import platform

from types import ModuleType

from ..version import __version__

__all__ = [
    'try_import_mxboard',
    'try_import_mxnet',
    'try_import_catboost',
    'try_import_lightgbm',
    'try_import_xgboost',
    'try_import_faiss',
    'try_import_fastai',
    'try_import_torch',
    'try_import_d8',
    'try_import_autogluon_multimodal',
    'try_import_rapids_cuml',
    'try_import_imodels',
]

logger = logging.getLogger(__name__)


def try_import_mxboard():
    try:
        import mxboard
    except ImportError:
        raise ImportError(
            "Unable to import dependency mxboard. "
            "A quick tip is to install via `pip install mxboard`. ")


def try_import_mxnet():
    mx_version = '1.6.0'
    try:
        import mxnet as mx
        from distutils.version import LooseVersion

        if LooseVersion(mx.__version__) < LooseVersion(mx_version):
            msg = (
                "Legacy mxnet=={} detected, some new modules will not work properly. "
                "mxnet>={} is required. You can use pip to upgrade mxnet "
                "`pip install mxnet --upgrade` "
                "or `pip install mxnet_cu101 --upgrade`").format(mx.__version__, mx_version)
            raise ValueError(msg)
    except ImportError:
        raise ImportError(
            "Unable to import dependency mxnet. "
            "A quick tip is to install via `pip install mxnet --upgrade`, "
            "or `pip install mxnet_cu101 --upgrade`")


def try_import_ray() -> ModuleType:
    RAY_MAX_VERSION = '2.3.0'
    ray_max_version_os_map = dict(
        Darwin=RAY_MAX_VERSION,
        Windows=RAY_MAX_VERSION,
        Linux=RAY_MAX_VERSION,
    )
    ray_min_version = '2.2.0'
    current_os = platform.system()
    ray_max_version = ray_max_version_os_map.get(current_os, RAY_MAX_VERSION)
    try:
        import ray
        from distutils.version import LooseVersion

        if LooseVersion(ray.__version__) < LooseVersion(ray_min_version) or LooseVersion(ray.__version__) >= LooseVersion(ray_max_version):
            msg = (
                f"ray=={ray.__version__} detected. "
                f"{ray_min_version} <= ray < {ray_max_version} is required. You can use pip to install certain version of ray "
                f"`pip install ray=={ray_min_version}` "
            )
            raise ValueError(msg)
        return ray
    except ImportError:
        raise ImportError(
            "ray is required to train folds in parallel. "
            f"A quick tip is to install via `pip install ray=={ray_min_version}`, "
            "or use sequential fold fitting by passing `sequential_local` to `ag_args_ensemble` when calling tabular.fit"
            "For example: `predictor.fit(..., ag_args_ensemble={'fold_fitting_strategy': 'sequential_local'})`"
        )
        
        
def try_import_ray_lightning():
    """This function tries to import ray lightning and check if the compatible pytorch lightning version is installed"""
    supported_ray_lightning_min_version = '0.2.0'
    supported_ray_lightning_max_version = '0.3.0'
    ray_lightning_torch_lightning_compatibility_map = {
        '0.2.x': '1.5.x',
    }
    ray_lightining_torch_lightning_compatibility_range_map = {
        ('0.2.0', '0.3.0'): ('1.5.0', '1.6.0'),
    }
    try:
        import pkg_resources
        import pytorch_lightning
        import ray_lightning
        from distutils.version import LooseVersion
        
        ray_lightning_version = pkg_resources.get_distribution('ray_lightning').version # ray_lightning doesn't have __version__...
        
        if not (LooseVersion(supported_ray_lightning_min_version)
                <= LooseVersion(ray_lightning_version)
                < LooseVersion(supported_ray_lightning_max_version)):
            logger.log(
                f"ray_lightning=={ray_lightning_version} detected. "
                f"{supported_ray_lightning_min_version} <= ray_lighting < {supported_ray_lightning_max_version} is required."
                "You can use pip to install certain version of ray_lightning."
                f"Supported ray_lightning versions and the compatible torch lightning versions are {ray_lightning_torch_lightning_compatibility_map}."
            )
            return False
        
        for ray_lightning_versions, torch_lightning_versions in ray_lightining_torch_lightning_compatibility_range_map.items():
            ray_lightning_min_version, ray_lightning_max_version = ray_lightning_versions
            torch_lightning_min_version, torch_lightning_max_version = torch_lightning_versions
            if (
                LooseVersion(ray_lightning_min_version)
                <= LooseVersion(ray_lightning_version)
                < LooseVersion(ray_lightning_max_version)
            ):
                if not (LooseVersion(torch_lightning_min_version)
                        <= LooseVersion(pytorch_lightning.__version__)
                        < LooseVersion(torch_lightning_max_version)):
                    logger.log(
                        f"Found ray_lightning {ray_lightning_version} that's not compatible with pytorch_lightning."
                        f"The compatible version of pytorch_lightning is >= {torch_lightning_min_version} and < {torch_lightning_max_version}."
                    )
                    return False
        return True
            

    except ImportError:
        logger.info(
            "You can enable each individual trial using multiple gpus by installing ray_lightning."
            f"Supported ray_lightning versions and the compatible torch lightning versions are {ray_lightning_torch_lightning_compatibility_map}."
        )
        return False


def try_import_catboost():
    try:
        import catboost
    except ImportError as e:
        raise ImportError("`import catboost` failed. "
                          f"A quick tip is to install via `pip install autogluon.tabular[catboost]=={__version__}`.")
    except ValueError as e:
        raise ImportError("Import catboost failed. Numpy version may be outdated, "
                          "Please ensure numpy version >=1.17.0. If it is not, please try 'pip uninstall numpy -y; pip install numpy>=1.17.0' "
                          "Detailed info: {}".format(str(e)))


def try_import_lightgbm():
    try:
        import lightgbm
    except ImportError as e:
        raise ImportError("`import lightgbm` failed. "
                          f"A quick tip is to install via `pip install autogluon.tabular[lightgbm]=={__version__}`.")
    except OSError as e:
        raise ImportError("`import lightgbm` failed. If you are using Mac OSX, "
                          "Please try 'brew install libomp'. Detailed info: {}".format(str(e)))


def try_import_xgboost():
    try:
        import xgboost
        from pkg_resources import parse_version  # pylint: disable=import-outside-toplevel
        xgboost_version = parse_version(xgboost.__version__)
        min_version = "1.6"
        assert xgboost_version >= parse_version(min_version),\
            f'Currently, we only support "xgboost>={min_version}". Installed version: "xgboost=={xgboost.__version__}".'
    except ImportError:
        raise ImportError("`import xgboost` failed. "
                          f"A quick tip is to install via `pip install autogluon.tabular[xgboost]=={__version__}`.")


def try_import_faiss():
    try:
        import faiss
    except ImportError:
        raise ImportError(
            "Unable to import dependency faiss. "
            "A quick tip is to install via `pip install faiss-cpu`. ")


def try_import_fastai():
    try:
        from pkg_resources import parse_version  # pylint: disable=import-outside-toplevel
        import fastai
        fastai_version = parse_version(fastai.__version__)
        assert parse_version('2.0.0') <= fastai_version < parse_version('2.8'), 'Currently, we only support 2.0.0<=fastai<2.8'

        # fastai is doing library setup during star imports. These are required for correct library functioning.
        # Local star imports are not possible in-place, so separate helper packages is created
        import autogluon.tabular.models.fastainn.imports_helper

    except ModuleNotFoundError as e:
        raise ImportError(f"Import fastai failed. A quick tip is to install via `pip install autogluon.tabular[fastai]=={__version__}`. ")


def try_import_torch():
    try:
        import torch
    except ImportError as e:
        raise ImportError("Unable to import dependency torch\n"
                          "A quick tip is to install via `pip install torch`.\n"
                          "The minimum torch version is currently 1.6.")


def try_import_d8():
    try:
        import d8
    except ImportError as e:
        raise ImportError("`import d8` failed. d8 is an optional dependency.\n"
                          "A quick tip is to install via `pip install d8`.\n")


def try_import_autogluon_multimodal():
    try:
        import autogluon.multimodal
    except ImportError:
        raise ImportError("`import autogluon.multimodal` failed.\n"
                          f"A quick tip is to install via `pip install autogluon.multimodal=={__version__}`.\n")


def try_import_rapids_cuml():
    try:
        import cuml
    except ImportError:
        raise ImportError("`import cuml` failed.\n"
                          "Ensure that you have a GPU and CUDA installation, and then install RAPIDS.\n"
                          "You will likely need to create a fresh conda environment based off of a RAPIDS install, and then install AutoGluon on it.\n"
                          "RAPIDS is highly experimental within AutoGluon, and we recommend to only use RAPIDS if you are an advanced user / developer.\n"
                          "Please refer to RAPIDS install instructions for more information: https://rapids.ai/start.html#get-rapids")


def try_import_imodels():
    try:
        import imodels
    except ImportError:
        raise ImportError(
            "Unable to import dependency imodels. "
            "A quick tip is to install via `pip install imodels`. ")


def try_import_vowpalwabbit():
    try:
        import vowpalwabbit
        from pkg_resources import parse_version  # pylint: disable=import-outside-toplevel
        vowpalwabbit_version = parse_version(vowpalwabbit.__version__)
        assert vowpalwabbit_version >= parse_version('8.10.1') and vowpalwabbit_version < parse_version('8.11.0'), \
            f'Currently, we only support VW version >=8.10.1 and <8.11.0. Found vowpalwabbit version: {vowpalwabbit_version}'
    except ImportError:
        raise ImportError("`import vowpalwabbit` failed.\n"
                          "A quick tip is to install via `pip install vowpalwabbit==8.10.1")
