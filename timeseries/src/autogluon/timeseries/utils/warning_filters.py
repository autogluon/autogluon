import contextlib
import functools
import io
import logging
import os
import sys
import warnings

from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning

__all__ = ["warning_filter", "disable_root_logger", "disable_tqdm"]


@contextlib.contextmanager
def warning_filter():
    with warnings.catch_warnings():
        env_py_warnings = os.environ.get("PYTHONWARNINGS", "")
        for warning_category in [RuntimeWarning, UserWarning, ConvergenceWarning, ValueWarning, FutureWarning]:
            warnings.simplefilter("ignore", category=warning_category)
        try:
            os.environ["PYTHONWARNINGS"] = "ignore"
            yield
        finally:
            os.environ["PYTHONWARNINGS"] = env_py_warnings


@contextlib.contextmanager
def disable_root_logger():
    try:
        logging.getLogger().setLevel(logging.ERROR)
        yield
    finally:
        logging.getLogger().setLevel(logging.INFO)


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
