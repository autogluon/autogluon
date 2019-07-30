__all__ = ['try_import_mxboard', 'try_import_mxnet', 'try_import_dask', 'try_import_cv2']

def try_import_mxboard():
    try:
        import mxboard
    except ImportError:
        raise ImportError(
            "Unable to import dependency mxboard. "
            "A quick tip is to install via `pip install mxboard`. ")

def try_import_mxnet():
    mx_version = '1.4.1'
    try:
        import mxnet as mx
        from distutils.version import LooseVersion

        if LooseVersion(mx.__version__) < LooseVersion(mx_version):
            msg = (
                "Legacy mxnet-mkl=={} detected, some new modules may not work properly. "
                "mxnet-mkl>={} is required. You can use pip to upgrade mxnet "
                "`pip install mxnet-mkl --pre --upgrade` "
                "or `pip install mxnet-cu90mkl --pre --upgrade`").format(mx.__version__, mx_version)
            raise ImportError(msg)
    except ImportError:
        raise ImportError(
            "Unable to import dependency mxnet. "
            "A quick tip is to install via `pip install mxnet-mkl/mxnet-cu90mkl --pre`. ")

def try_import_dask():
    try:
        import dask
    except ImportError:
        raise ImportError(
            "Unable to import dependency dask. "
            "A quick tip is to install via `pip install dask[complete]`. ")

def try_import_cv2():
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "Unable to import dependency cv2. "
            "A quick tip is to install via `pip install opencv-python`. ")
