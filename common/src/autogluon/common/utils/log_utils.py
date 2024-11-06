import logging
from typing import Optional

_logger_ag = logging.getLogger("autogluon")  # return autogluon root logger


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
    """Translates verbosity to logging level. Suppresses warnings if verbosity = 0."""
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


def add_log_to_file(file_path: str, logger: Optional[logging.Logger] = None):
    """
    Add a FileHandler to the logger so that it can log to a file

    Parameters
    ----------
    file_path: str
        File path to save the log
    logger: Optional[logging.Logger], default = None
        The log to add FileHandler.
        If not provided, will add to the default AG logger, `logging.getLogger('autogluon')`
    """
    if logger is None:
        logger = _logger_ag
    fh = logging.FileHandler(file_path)
    logger.addHandler(fh)


def _check_if_kaggle() -> bool:
    """
    Returns True if inside Kaggle Notebook
    """
    root_logger = logging.getLogger()
    for handler in root_logger.root.handlers[:]:
        if hasattr(handler, "baseFilename") and (handler.baseFilename == "/tmp/kaggle.log"):  # type: ignore
            return True
    return False


def _add_stream_handler():
    # Add stream_handler to AG logger if it doesn't already exist
    # This is necessary so that the modification of logging level can take effect
    # Also this adjust the logging format
    # This function is supposed to be called before any logging from autogluon happens
    if not any(isinstance(h, logging.StreamHandler) for h in _logger_ag.handlers):
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter("%(message)s")
        stream_handler.setFormatter(formatter)
        _logger_ag.addHandler(stream_handler)
        _logger_ag.propagate = False


__FIXED_KAGGLE_LOGGING = False
__FIXED_SKLEARNEX_LOGGING = False


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


def fix_sklearnex_logging_if_kaggle():
    """
    Fixes logging verbosity for sklearnex when in a Kaggle notebook.
    By default, sklearnex verbosity is set to `info` in Kaggle, which results in unintended logging spam.
    This corrects this by detected if we are in a Kaggle environment and then setting the logger verbosity back to WARNING.

    For more details, refer to the following:
        1. https://github.com/intel/scikit-learn-intelex/issues/1695#issuecomment-1948647937
        2. https://github.com/autogluon/autogluon/issues/4141
    """
    global __FIXED_SKLEARNEX_LOGGING
    if (not __FIXED_SKLEARNEX_LOGGING) and _check_if_kaggle():
        logging.getLogger("sklearnex").setLevel("WARNING")
    # After the fix is performed, no need to fix again.
    __FIXED_SKLEARNEX_LOGGING = True


def convert_time_in_s_to_log_friendly(time_in_sec: float, min_value: float = 0.01):
    """
    Converts a time in seconds to a logging friendly version with updated units.

    Parameters
    ----------
    time_in_sec : float
        The original time in seconds to convert.
    min_value : float, default = 0.01
        The minimum value time_adjusted should be.
        If the value is greater than this, it will use a smaller time_unit until the value is greater than min_value or the smallest time_unit is reached.

    Returns
    -------
    Returns a tuple of time_adjusted: float, time_unit: str that is the log friendly version of the time with corresponding time unit.

    """
    values = [
        ("s", 1),
        ("ms", 1e3),
        ("μs", 1e6),
        ("ns", 1e9),
    ]
    time_adjusted = time_in_sec
    time_unit = "s"
    for time_unit, time_factor in values:
        time_adjusted = time_in_sec * time_factor
        if time_adjusted >= min_value:
            break
    return time_adjusted, time_unit
