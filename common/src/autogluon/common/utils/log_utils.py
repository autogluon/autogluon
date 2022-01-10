import logging

_logger_ag = logging.getLogger('autogluon')  # return autogluon root logger


class DuplicateFilter(object):
    """Filter duplicate log messages based on filter_targets

    Example usage:
        dup_filter = DuplicateFilter(['a'])
        logger.addFilter(dup_filter)
        for i in range(10):
            logger.info('a') # will only log once
            logger.info('abc') # will log 10 times
        dup_filter.attach_filter_targets('abc')
        for i in range(10):
            logger.info('abc') # will only log once now
        dup_filter.clear_filter_targets() # nothing filtered anymore
    """
    def __init__(self, filter_targets=[]):
        self.msgs = set()
        self.filter_targets = set(filter_targets)

    def filter(self, record):
        rv = record.msg not in self.msgs
        if record.msg in self.filter_targets:
            self.msgs.add(record.msg)
        return rv

    def attach_filter_targets(self, filter_targets):
        if type(filter_targets) == str:
            filter_targets = [filter_targets]
        for target in filter_targets:
            self.filter_targets.add(target)

    def clear_filter_targets(self):
        self.msgs = set()
        self.filter_targets = set()


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
        logger = _logger_ag
    if verbosity < 0:
        verbosity = 0
    elif verbosity > 4:
        verbosity = 4
    logger.setLevel(verbosity2loglevel(verbosity))


def _check_if_kaggle() -> bool:
    """
    Returns True if inside Kaggle Notebook
    """
    root_logger = logging.getLogger()
    for handler in root_logger.root.handlers[:]:
        if hasattr(handler, 'baseFilename') and (handler.baseFilename == '/tmp/kaggle.log'):
            return True
    return False


def _add_stream_handler():
    stream_handler = logging.StreamHandler()
    # add stream_handler to AG logger
    _logger_ag.addHandler(stream_handler)


__FIXED_KAGGLE_LOGGING = False


def fix_logging_if_kaggle():
    """
    Fixes logger in Kaggle. In Kaggle logging is redirected to a file which hides all AutoGluon log output from the notebook.
    This function checks if we are in a Kaggle notebook, and if so adds a StreamHandler to AutoGluon's logger to ensure logs are shown.
    """
    global __FIXED_KAGGLE_LOGGING
    if (not __FIXED_KAGGLE_LOGGING) and _check_if_kaggle():
        _add_stream_handler()
    # After the fix is performed, or it is determined we are not in Kaggle, no need to fix again.
    __FIXED_KAGGLE_LOGGING = True
