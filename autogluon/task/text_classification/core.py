import logging
from typing import Any
__all__ = []

logger = logging.getLogger(__name__)


class Results(object):
    """
    Python class to hold the results for the trials
    """
    def __init__(self, model: Any, metric: Any, config:Any, time: int):
        self.model = model
        self.metric = metric
        self.config = config
        self.time = time
