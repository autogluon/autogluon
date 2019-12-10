import os
import warnings

__all__ = ['in_ipynb', 'warning_filter', 'verbosity2loglevel']

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

def verbosity2loglevel(verbosity):
    """ Translates verbosity to logging level. Suppresses warnings if verbosity = 0. """
    if verbosity <= 0: # only errors
        warnings.filterwarnings("ignore")
        # print("Caution: all warnings suppressed")
        log_level = 40
    elif verbosity == 1: # only warnings and critical print statements
        log_level = 25
    elif verbosity == 2: # key print statements which should be shown by default
        log_level = 20 
    elif verbosity == 3: # more-detailed printing
        log_level = 15
    else:
        log_level = 10 # print everything (ie. debug mode)
    
    return log_level