__all__ = ['try_import_mxnet', 'try_import_cv2', 'try_import_gluonnlp']


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
