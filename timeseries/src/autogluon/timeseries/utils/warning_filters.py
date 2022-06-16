import contextlib
import logging
import os
import warnings

__all__ = ["evaluator_warning_filter", "disable_root_logger"]


@contextlib.contextmanager
def evaluator_warning_filter():
    env_py_warnings = os.environ.get("PYTHONWARNINGS", "")
    warning_categories = [RuntimeWarning, UserWarning, FutureWarning]  # ignore these
    try:
        for warning_category in warning_categories:
            warnings.filterwarnings("ignore", category=warning_category)

        # required to suppress gluonts evaluation warnings as the module uses multiprocessing
        os.environ["PYTHONWARNINGS"] = ",".join([f"ignore::{c.__name__}" for c in warning_categories])

        yield
    finally:
        for warning_category in warning_categories:
            warnings.filterwarnings("default", category=warning_category)
        os.environ["PYTHONWARNINGS"] = env_py_warnings


@contextlib.contextmanager
def disable_root_logger():
    try:
        logging.getLogger().setLevel(logging.ERROR)
        yield
    finally:
        logging.getLogger().setLevel(logging.INFO)
