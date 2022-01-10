import logging

logger = logging.getLogger(__name__)

__all__ = ['FakeReporter']


class FakeReporter(object):
    """FakeReporter for internal use in final fit
    """
    def __call__(self, **kwargs):
        pass
