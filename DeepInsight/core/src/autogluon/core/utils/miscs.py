import logging
import os
import warnings

__all__ = ['in_ipynb', 'warning_filter', 'verbosity2loglevel', 'set_logger_verbosity']

_logger = logging.getLogger()  # return root logger


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


class warning_filter(warnings.catch_warnings):
    def __enter__(self):
        super().__enter__()
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        return self


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


def set_logger_verbosity(verbosity: int, logger=None):
    if logger is None:
        logger = _logger
    if verbosity < 0:
        verbosity = 0
    elif verbosity > 4:
        verbosity = 4
    logger.setLevel(verbosity2loglevel(verbosity))
