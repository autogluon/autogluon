import logging


def verbosity2loglevel(verbosity):
    """ Translates verbosity to logging level. Suppresses warnings if verbosity = 0. """
    if verbosity <= 0:  # only critical
        log_level = logging.ERROR
    elif verbosity == 1:  # only error and critical
        log_level = logging.WARNING
    elif verbosity == 2:  # warning, error and critical
        log_level = logging.INFO
    elif verbosity == 3:  # info, warning, error and critical
        log_level = logging.DEBUG
    else:
        log_level = logging.NOTSET  # print everything (ie. debug mode)

    return log_level


def set_logger_verbosity(verbosity: int):
    if verbosity < 0:
        verbosity = 0
    elif verbosity > 4:
        verbosity = 4
    logging.disable(verbosity2loglevel(verbosity))
