
import warnings

__all__ = ['verbosity2loglevel']


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
