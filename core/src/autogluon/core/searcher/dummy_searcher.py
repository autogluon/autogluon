import logging

from .local_searcher import LocalSearcher
from .exceptions import ExhaustedSearchSpaceError

__all__ = ['DummySearcher']

logger = logging.getLogger(__name__)


class DummySearcher(LocalSearcher):
    """Searcher which only returns the default config."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._exhausted = False

    def get_config(self, **kwargs) -> dict:
        if self._exhausted:
            raise ExhaustedSearchSpaceError('Default config already provided. Search space is exhausted!')
        self._exhausted = True
        return self._params_default
