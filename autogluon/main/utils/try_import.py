__all__ = ['try_import_catboost', 'try_import_lightgbm', 'try_import_mxboard', 'try_import_mxnet',
           'try_import_cv2', 'try_import_gluonnlp', 'try_import_fastai_v1']

def try_import_catboost():
    try:
        import catboost
    except ValueError as e:
        raise ImportError("Import catboost failed. Numpy version may be outdated, "
                          "Please ensure numpy version >=1.16.0. If it is not, please try 'pip uninstall numpy; pip install numpy>=1.17.0' Detailed info: {}".format(str(e)))

def try_import_catboostdev():  # TODO: remove once Catboost 0.24 is released.
    try:
        import catboost  # Need to first import catboost before catboost_dev and not vice-versa
        import catboost_dev
    except (ValueError, ImportError) as e:
        raise ImportError("Import catboost_dev failed (needed for distillation with CatBoost models). "
                          "Make sure you can import catboost and then run: 'pip install catboost-dev'."
                          "Detailed info: {}".format(str(e)))

def try_import_lightgbm():
    try:
        import lightgbm
    except OSError as e:
        raise ImportError("Import lightgbm failed. If you are using Mac OSX, "
                          "Please try 'brew install libomp'. Detailed info: {}".format(str(e)))

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

def try_import_faiss():
    try:
        import faiss
    except ImportError:
        raise ImportError(
            "Unable to import dependency faiss"
            "A quick tip is to install via `pip install faiss-cpu`. ")

def try_import_fastai_v1():
    try:
        from pkg_resources import parse_version  # pylint: disable=import-outside-toplevel
        import fastai
        fastai_version = parse_version(fastai.__version__)
        assert parse_version('1.0.61') <= fastai_version < parse_version('2.0.0'), 'Currently, we only support 1.0.61<=fastai<2.0.0'
    except ModuleNotFoundError as e:
        raise ImportError("Import fastai failed. A quick tip is to install via `pip install fastai==1.*`. "
                          "If you are using Mac OSX, please use this torch version to avoid compatibility issues: `pip install torch==1.6.0`.")
