import contextlib
import functools
import logging
import os

__all__ = ["evaluator_warning_filter", "disable_root_logger", "disable_tqdm"]


@contextlib.contextmanager
def evaluator_warning_filter():
    env_py_warnings = os.environ.get("PYTHONWARNINGS", "")
    warning_categories = [RuntimeWarning, UserWarning, FutureWarning]  # ignore these
    try:
        # required to suppress gluonts evaluation warnings as the module uses multiprocessing
        os.environ["PYTHONWARNINGS"] = ",".join([f"ignore::{c.__name__}" for c in warning_categories])
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
