import os
import sys
import time
import json
import logging
import threading
import multiprocessing as mp
from ..utils import save, load, AutoGluonEarlyStop
import distributed
from distributed import Queue, Variable
from distributed.comm.core import CommClosedError

logger = logging.getLogger(__name__)

__all__ = ['DistStatusReporter', 'FakeReporter', 'DistSemaphore']


class FakeReporter(object):
    """FakeReporter for internal use in final fit
    """
    def __call__(self, **kwargs):
        pass


class DistStatusReporter(object):
    """Report status through the training scheduler.

    Example:
        >>> @autogluon_method
        >>> def train_func(config, reporter):
        >>>     reporter(accuracy=0.1)
    """

    def __init__(self, remote=None):
        self._queue = Queue(client=remote)
        self._stop = Variable(client=remote)
        self._stop.set(False)
        self._continue_semaphore = DistSemaphore(0, remote)
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

        logger.debug('Reporting {}'.format(json.dumps(kwargs)))
        try:
            self._queue.put(kwargs.copy())
        except RuntimeError:
            return
        self._continue_semaphore.acquire()
        if self._stop.get():
            raise AutoGluonEarlyStop('Stopping!')

    def fetch(self, block=True):
        try:
            kwargs = self._queue.get()
        except CommClosedError:
            return {}
        return kwargs

    def terminate(self):
        self._stop.set(True)
        self._continue_semaphore.release()

    def move_on(self):
        self._continue_semaphore.release()

    def _start(self):
        """Adjust the real starting time
        """
        self._last_report_time = time.time()

    def save_dict(self, **state_dict):
        raise NotImplementedError

    def get_dict(self):
        raise NotImplementedError

    def __repr__(self):
        reprstr = self.__class__.__name__
        return reprstr


class DistSemaphore(object):
    def __init__(self, value, remote=None):
        self._queue = Queue(client=remote)
        for i in range(value):
            self._queue.put(1)

    def acquire(self):
        try:
            _ = self._queue.get()
        except distributed.comm.core.CommClosedError:
            pass

    def release(self):
        self._queue.put(1)

    def __repr__(self):
        reprstr = self.__class__.__name__
        return reprstr
