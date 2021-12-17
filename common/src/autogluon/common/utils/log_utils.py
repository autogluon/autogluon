import logging

_logger = logging.getLogger('autogluon')  # return autogluon root logger


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
        logger = _logger
    if verbosity < 0:
        verbosity = 0
    elif verbosity > 4:
        verbosity = 4
    logger.setLevel(verbosity2loglevel(verbosity))
