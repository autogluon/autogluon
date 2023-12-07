import warnings

__all__ = ["warning_filter"]


class warning_filter(warnings.catch_warnings):
    def __enter__(self):
        super().__enter__()
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        return self
