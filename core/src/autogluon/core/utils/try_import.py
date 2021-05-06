__all__ = [
    'try_import_mxboard',
    'try_import_mxnet',
    'try_import_catboost',
    'try_import_catboostdev',
    'try_import_lightgbm',
    'try_import_xgboost',
    'try_import_faiss',
    'try_import_fastai',
    'try_import_cv2',
    'try_import_gluonnlp',
    'try_import_torch',
    'try_import_skopt',
    'try_import_autogluon_text',
    'try_import_autogluon_vision',
    'try_import_rapids_cuml']


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


def try_import_catboost():
    try:
        import catboost
    except ImportError as e:
        raise ImportError("`import catboost` failed."
                          "A quick tip is to install via `pip install catboost`.")
    except ValueError as e:
        raise ImportError("Import catboost failed. Numpy version may be outdated, "
                          "Please ensure numpy version >=1.16.0. If it is not, please try 'pip uninstall numpy -y; pip install numpy>=1.17.0' "
                          "Detailed info: {}".format(str(e)))


def try_import_catboostdev():  # TODO: remove once Catboost 0.24 is released.
    try:
        import catboost  # Need to first import catboost before catboost_dev and not vice-versa
        import catboost_dev
    except (ValueError, ImportError) as e:
        raise ImportError("`import catboost_dev` failed (needed for distillation with CatBoost models). "
                          "Make sure you can import catboost and then run: 'pip install catboost-dev'."
                          "Detailed info: {}".format(str(e)))


def try_import_lightgbm():
    try:
        import lightgbm
    except ImportError as e:
        raise ImportError("`import lightgbm` failed. "
                          "A quick tip is to install via `pip install lightgbm`.")
    except OSError as e:
        raise ImportError("`import lightgbm` failed. If you are using Mac OSX, "
                          "Please try 'brew install libomp'. Detailed info: {}".format(str(e)))


def try_import_xgboost():
    try:
        import xgboost
    except ImportError:
        raise ImportError("`import xgboost` failed. "
                          "A quick tip is to install via `pip install xgboost`.")


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
        assert parse_version('2.0.0') <= fastai_version < parse_version('3.0.0'), 'Currently, we only support 2.0.0<=fastai<3.0.0'

        # fastai is doing library setup during star imports. These are required for correct library functioning.
        # Local star imports are not possible in-place, so separate helper packages is created
        import autogluon.tabular.models.fastainn.imports_helper

    except ModuleNotFoundError as e:
        raise ImportError("Import fastai failed. A quick tip is to install via `pip install fastai==2.*`. "
                          "If you are using Mac OSX, please use this torch version to avoid compatibility issues: `pip install torch==1.6.0`.")


def try_import_cv2():
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "Unable to import dependency cv2. "
            "A quick tip is to install via `pip install opencv-python`. ")


def try_import_gluonnlp():
    try:
        import gluonnlp
        # TODO After 1.0 is supported,
        #  we will remove the checking here and use gluonnlp.utils.check_version instead.
        from pkg_resources import parse_version  # pylint: disable=import-outside-toplevel
        gluonnlp_version = parse_version(gluonnlp.__version__)
        assert gluonnlp_version >= parse_version('0.8.1') and\
               gluonnlp_version <= parse_version('0.8.3'), \
            'Currently, we only support 0.8.1<=gluonnlp<=0.8.3'
    except ImportError:
        raise ImportError(
            "Unable to import dependency gluonnlp. The NLP model won't be available "
            "without installing gluonnlp. "
            "A quick tip is to install via `pip install gluonnlp==0.8.1`. ")
    return gluonnlp


def try_import_torch():
    try:
        import torch
    except ImportError as e:
        raise ImportError("Unable to import dependency torch\n"
                          "A quick tip is to install via `pip install torch`.\n"
                          "The minimum torch version is currently 1.6.")


def try_import_skopt():
    try:
        import skopt
    except ImportError:
        raise ImportError("`import skopt` failed. skopt is an optional dependency and may not be installed.\n"
                          "A quick tip is to install via `pip install scikit-optimize`.")


def try_import_autogluon_text():
    try:
        import autogluon.text
    except ImportError:
        raise ImportError("`import autogluon.text` failed.\n"
                          "A quick tip is to install via `pip install autogluon.text`.\n"
                          "Ensure that the version installed is the same as the version of the other autogluon modules seen in `pip freeze`.")


def try_import_autogluon_vision():
    try:
        import autogluon.vision
    except ImportError:
        raise ImportError("`import autogluon.vision` failed.\n"
                          "A quick tip is to install via `pip install autogluon.vision`.\n"
                          "Ensure that the version installed is the same as the version of the other autogluon modules seen in `pip freeze`.")


def try_import_rapids_cuml():
    try:
        import cuml
    except ImportError:
        raise ImportError("`import cuml` failed.\n"
                          "Ensure that you have a GPU and CUDA installation, and then install RAPIDS.\n"
                          "You will likely need to create a fresh conda environment based off of a RAPIDS install, and then install AutoGluon on it.\n"
                          "RAPIDS is highly experimental within AutoGluon, and we recommend to only use RAPIDS if you are an advanced user / developer.\n"
                          "Please refer to RAPIDS install instructions for more information: https://rapids.ai/start.html#get-rapids")
