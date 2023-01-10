from .utils import get_autogluon_metadata


def disable_if_lite_mode(ret=None):
    def inner(func):
        def do_nothing(*args, **kwargs):
            if callable(ret):
                return ret(*args, **kwargs)
            return ret
        metadata = get_autogluon_metadata()
        if 'lite' in metadata and metadata['lite']:
            return do_nothing
        return func
    return inner
