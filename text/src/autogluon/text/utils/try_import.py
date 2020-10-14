__all__ = ['try_import_gluonnlp']


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
