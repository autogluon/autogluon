import logging
import warnings

__all__ = ["evaluator_warning_filter", "serialize_warning_filter"]


class evaluator_warning_filter(object):
    def __enter__(self):
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        return self

    def __exit__(self, *args):
        warnings.filterwarnings("default", category=RuntimeWarning)
        warnings.filterwarnings("default", category=UserWarning)
        warnings.filterwarnings("default", category=FutureWarning)


class serialize_warning_filter(object):
    def __enter__(self):
        logging.getLogger().setLevel(logging.ERROR)
        return self

    def __exit__(self, *arg):
        logging.getLogger().setLevel(logging.INFO)
