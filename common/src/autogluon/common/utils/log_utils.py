import logging
import warnings

_logger = logging.getLogger('autogluon')  # return autogluon root logger


def verbosity2loglevel(verbosity):
    """ Translates verbosity to logging level. Suppresses warnings if verbosity = 0. """
    if verbosity <= 0:  # only errors
        # print("Caution: all warnings suppressed")
        log_level = 40
    elif verbosity == 1:  # only warnings and critical print statements
        log_level = 25
    elif verbosity == 2:  # key print statements which should be shown by default
        log_level = 20
    elif verbosity == 3:  # more-detailed printing
        log_level = 15
    else:
        log_level = 10  # print everything (ie. debug mode)

    return log_level


def set_logger_verbosity(verbosity: int, logger=None):
    if logger is None:
        logger = _logger
    if verbosity < 0:
        verbosity = 0
    elif verbosity > 4:
        verbosity = 4
    logger.setLevel(verbosity2loglevel(verbosity))
