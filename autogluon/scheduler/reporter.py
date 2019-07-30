import os
import time
import json
import logging
import multiprocessing as mp

from ..basic import save, load

logger = logging.getLogger(__name__)

class StatusReporter(object):
    """Report status through the training scheduler.
    Example:
        >>> def train_func(config, reporter):
        >>>     assert isinstance(reporter, StatusReporter)
        >>>     reporter(timesteps_this_iter=1)
    """

    def __init__(self, dict_path=None):#, result_queue, continue_semaphore):
        self._queue = mp.Queue(1)
        self._last_report_time = None
        self._continue_semaphore = mp.Semaphore(0)
        self._last_report_time = time.time()
        self._save_dict = False
        self.dict_path = dict_path

    def __call__(self, **kwargs):
        """Report updated training status.
        Pass in `done=True` when the training job is completed.
        Args:
            kwargs: Latest training result status.
        Example:
            >>> reporter(accuracy=1, training_iters=4)
        """
        report_time = time.time()
        if 'time_this_iter' not in kwargs:
            kwargs['time_this_iter'] = report_time - self._last_report_time
        self._last_report_time = report_time

        self._queue.put(kwargs.copy(), block=True)
        self._continue_semaphore.acquire()

        logger.debug('StatusReporter reporting: {}'.format(json.dumps(kwargs)))

    def fetch(self, block=True):
        kwargs = self._queue.get(block=block)
        return kwargs

    def move_on(self):
        self._continue_semaphore.release()

    def _start(self):
        """Adjust the real starting time
        """
        self._last_report_time = time.time()

    def save_dict(self, **state_dict):
        """Save the serializable state_dict
        """
        logger.debug('Saving the task dict to {}'.format(self.dict_path))
        save(state_dict, self.dict_path)

    def has_dict(self):
        logger.debug('has_dict {}'.format(os.path.isfile(self.dict_path)))
        return os.path.isfile(self.dict_path)

    def get_dict(self):
        return load(self.dict_path)

    def __repr__(self):
        reprstr = self.__class__.__name__
        return reprstr
