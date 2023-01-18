from .utils import LITE_MODE


def disable_if_lite_mode(ret=None):
    def inner(func):
        def do_nothing(*args, **kwargs):
            if callable(ret):
                return ret(*args, **kwargs)
            return ret
        if LITE_MODE:
            return do_nothing
        return func
    return inner
