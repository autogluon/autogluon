import time
import multiprocessing as mp
import logging
import json

logger = logging.getLogger(__name__)

class StatusReporter(object):
    """Report status through the training scheduler.
    Example:
        >>> def train_func(config, reporter):
        >>>     assert isinstance(reporter, StatusReporter)
        >>>     reporter(timesteps_this_iter=1)
    """

    def __init__(self):#, result_queue, continue_semaphore):
        self._queue = mp.Queue(1)
        self._last_report_time = None
        self._continue_semaphore = mp.Semaphore(1)
        self._last_report_time = time.time()

    def __call__(self, **kwargs):
        """Report updated training status.
        Pass in `done=True` when the training job is completed.
        Args:
            kwargs: Latest training result status.
        Example:
            >>> reporter(accuracy=1, training_iters=4)
        """
        logger.debug('StatusReporter reporting: {}'.format(json.dumps(kwargs)))

        report_time = time.time()
        if 'time_this_iter' not in kwargs:
            kwargs['time_this_iter'] = report_time - self._last_report_time
        self._last_report_time = report_time

        self._queue.put(kwargs.copy(), block=True)
        self._continue_semaphore.acquire()

    def fetch(self, block=True):
        kwargs = self._queue.get(block=block)
        self._continue_semaphore.release()
        return kwargs

    def _start(self):
        """Adjust the real starting time
        """
        self._last_report_time = time.time()
