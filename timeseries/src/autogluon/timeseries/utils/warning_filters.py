import contextlib
import functools
import io
import logging
import os
import re
import sys
import warnings

from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning

__all__ = ["warning_filter", "disable_root_logger", "disable_tqdm"]


@contextlib.contextmanager
def warning_filter(all_warnings: bool = False):
    categories = [RuntimeWarning, UserWarning, ConvergenceWarning, ValueWarning, FutureWarning]
    if all_warnings:
        categories.append(Warning)
    with warnings.catch_warnings():
        env_py_warnings = os.environ.get("PYTHONWARNINGS", "")
        for warning_category in categories:
            warnings.simplefilter("ignore", category=warning_category)
        try:
            os.environ["PYTHONWARNINGS"] = "ignore"
            yield
        finally:
            os.environ["PYTHONWARNINGS"] = env_py_warnings


@contextlib.contextmanager
def disable_root_logger(root_log_level=logging.ERROR):
    try:
        logging.getLogger().setLevel(root_log_level)
        yield
    finally:
        logging.getLogger().setLevel(logging.INFO)


@contextlib.contextmanager
def set_loggers_level(regex: str, level=logging.ERROR):
    log_levels = {}
    try:
        for logger_name in logging.root.manager.loggerDict:
            if re.match(regex, logger_name):
                log_levels[logger_name] = logging.getLogger(logger_name).level
                logging.getLogger(logger_name).setLevel(level)
        yield
    finally:
        for logger_name, level in log_levels.items():
            logging.getLogger(logger_name).setLevel(level)


@contextlib.contextmanager
def disable_tqdm():
    """monkey-patch tqdm to disable it within context"""
    try:
        from tqdm import tqdm

        _init = tqdm.__init__
        tqdm.__init__ = functools.partialmethod(tqdm.__init__, disable=True)
        yield
    except ImportError:
        yield
    else:
        tqdm.__init__ = _init


@contextlib.contextmanager
def disable_stdout():
    save_stdout = sys.stdout
    sys.stdout = io.StringIO()
    yield
    sys.stdout = save_stdout
