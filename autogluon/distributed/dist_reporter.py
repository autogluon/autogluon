import time
import json
import logging
import threading
from dask.distributed import Queue

logger = logging.getLogger(__name__)

__all__ = ['Communicator', 'DistStatusReporter']

class Communicator(threading.Thread):
    def __init__(self, process, local_reporter, dist_reporter):
        super(Communicator, self).__init__()
        self.process = process
        self.local_reporter = local_reporter
        self.dist_reporter = dist_reporter

    def run(self):
        while self.process.is_alive():
            reported_result = self.local_reporter.fetch()
            self.dist_reporter(**reported_result)
            self.local_reporter.move_on()
            if 'done' in reported_result and reported_result['done'] is True:
                self.process.join()
                break

    @classmethod
    def Create(cls, process, local_reporter, dist_reporter):
        communicator = cls(process, local_reporter, dist_reporter)
        communicator.start()
        return communicator

class DistStatusReporter(object):
    """Report status through the training scheduler.

    Example:
        >>> @autogluon_method
        >>> def train_func(config, reporter):
        >>>     reporter(accuracy=0.1)
    """

    def __init__(self):
        self._queue = Queue()
        self._continue_semaphore = DistSemaphore(1)
        self._last_report_time = time.time()

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

        #logger.debug('Reporting {}'.format(json.dumps(kwargs)))
        self._queue.put(kwargs.copy())
        self._continue_semaphore.acquire()

    def fetch(self, block=True):
        kwargs = self._queue.get()
        return kwargs

    def move_on(self):
        self._continue_semaphore.release()

    def _start(self):
        """Adjust the real starting time
        """
        self._last_report_time = time.time()


class DistSemaphore(object):
    def __init__(self, value):
        self._queue = Queue()
        for i in range(value):
            self._queue.put(1)

    def acquire(self):
        _ = self._queue.get()

    def release(self):
        self._queue.put(1)
