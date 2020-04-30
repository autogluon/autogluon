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

__all__ = ['DistStatusReporter', 'FakeReporter', 'DistSemaphore',
           'Communicator', 'LocalStatusReporter']


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


class LocalStatusReporter(object):
    """Local status reporter (automatically created by communicator)
    Example:
        >>> def train_func(config, reporter):
        >>>     assert isinstance(reporter, StatusReporter)
        >>>     reporter(timesteps_this_iter=1)
    """

    def __init__(self, dict_path=None):#, result_queue, continue_semaphore):
        self._queue = mp.Queue(1)
        self._stop = mp.Value('i', 0)
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
        logger.debug('StatusReporter reporting: {}'.format(json.dumps(kwargs)))

        self._continue_semaphore.acquire()
        if self._stop.value:
            raise AutoGluonEarlyStop

    def fetch(self, block=True):
        kwargs = self._queue.get(block=block)
        return kwargs

    def move_on(self):
        self._continue_semaphore.release()

    def terminate(self):
        self._stop.value = 1
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


class Communicator(threading.Thread):
    def __init__(self, process, local_reporter, dist_reporter):
        super().__init__()
        self.process = process
        self.local_reporter = local_reporter
        self.dist_reporter = dist_reporter
        self._stop_event = threading.Event()

    def run(self):
        while self.process.is_alive():

            # breaking communication if process raises exception
            if self.process.exception is not None:
                error, traceback = self.process.exception
                self.local_reporter.terminate()
                self.dist_reporter(done=True, traceback=traceback)
                self.process.join()
                break

            try:
                # waiting until process reports results or raises exception
                if self.local_reporter._queue.empty():
                    continue
                reported_result = self.local_reporter.fetch()
            except BrokenPipeError:
                break

            try:
                self.dist_reporter(**reported_result)
                self.local_reporter.move_on()
            except AutoGluonEarlyStop:
                self.local_reporter.terminate()

            if reported_result.get('done', False):
                self.process.join()
                break

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    @classmethod
    def Create(cls, process, local_reporter, dist_reporter):
        communicator = cls(process, local_reporter, dist_reporter)
        communicator.start()
        return communicator

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
