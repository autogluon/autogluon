import os
import warnings

__all__ = ['in_ipynb', 'warning_filter']

def in_ipynb():
    if 'AG_DOCS' in os.environ and os.environ['AG_DOCS']:
        return False
    try:
        cfg = get_ipython().config 
        if 'IPKernelApp' in cfg:
            return True
        else:
            return False
    except NameError:
        return False

class warning_filter(object):
    def __enter__(self):
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        return self

    def __exit__(self, *args):
        warnings.filterwarnings("default", category=UserWarning)
        warnings.filterwarnings("default", category=FutureWarning)
        warnings.filterwarnings("default", category=DeprecationWarning)
