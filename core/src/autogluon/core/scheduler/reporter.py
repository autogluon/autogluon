import os
import time
import json
import logging
import threading
import numpy as np
import multiprocessing as mp
from ..utils import save, load, AutoGluonEarlyStop
import distributed
from distributed import Queue, Variable
from distributed.comm.core import CommClosedError

logger = logging.getLogger(__name__)

__all__ = ['DistStatusReporter',
           'FakeReporter',
           'DistSemaphore',
           'Communicator',
           'LocalStatusReporter']


class FakeReporter(object):
    """FakeReporter for internal use in final fit
    """
    def __call__(self, **kwargs):
        pass


class DistStatusReporter(object):
    """Report status through the training scheduler.

    Example
    -------
    >>> @autogluon_method
    >>> def train_func(config, reporter):
    ...     reporter(accuracy=0.1)
    """

    def __init__(self, remote=None):
        self._queue = Queue(client=remote)
        self._stop = Variable(client=remote)
        self._stop.set(False)
        self._continue_semaphore = DistSemaphore(0, remote)

    def __call__(self, **kwargs):
        """Report updated training status.
        Pass in `done=True` when the training job is completed.

        Args:
            kwargs: Latest training result status.

        Example
        _______
        >>> reporter(accuracy=1, training_iters=4)
        """
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

    def save_dict(self, **state_dict):
        raise NotImplementedError

    def get_dict(self):
        raise NotImplementedError

    def __repr__(self):
        reprstr = self.__class__.__name__
        return reprstr


class MODistStatusReporter(DistStatusReporter):
    """Report status through the training of a multi-objective scheduler.

    Example
    -------
    >>> @autogluon_method
    >>> def train_func(config, reporter):
    ...     reporter(accuracy=1, f_score=1, training_iters=4)
    """
    def __init__(self, objectives, weights, scalarization_opts, remote=None):
        super().__init__(remote)
        self.objectives = objectives
        self.weights = weights
        self.scalarization_options = scalarization_opts
    
    def __call__(self, **kwargs):
        """Report updated training status.
        Pass in `done=True` when the training job is completed.

        Args:
            kwargs: Latest training result status. Reporter requires access to
            all objectives of interest.

        Example
        _______
        >>> reporter(accuracy=1, f_score=1, training_iters=4)
        """
        try:
            v = np.array([kwargs[k] for k in self.objectives])
        except KeyError:
            raise KeyError("Reporter requires accesss to all objective values.\
                Please ensure you return all required values.")

        if self.scalarization_options["algorithm"] == "random_weights":
            scalarization = max([w @ v for w in self.weights])
        elif self.scalarization_options["algorithm"] == "parego":
            rho = self.scalarization_options["rho"]
            scalarization = [max(w * v) + rho * (w @ v) for w in self.weights]
            scalarization = max(scalarization)
        else:
            raise ValueError("Specified scalarization algorithm is unknown. \
                Valid algorithms are 'random_weights' and 'parego'.")
        kwargs["_SCALARIZATION"] = scalarization

        super().__call__(**kwargs)


class LocalStatusReporter(object):
    """Local status reporter (automatically created by communicator)
    Example
    -------
    >>> def train_func(config, reporter):
    ...     assert isinstance(reporter, StatusReporter)
    ...     reporter(timesteps_this_iter=1)
    """

    def __init__(self, dict_path=None):#, result_queue, continue_semaphore):
        self._queue = mp.Queue(1)
        self._stop = mp.Value('i', 0)
        self._continue_semaphore = mp.Semaphore(0)
        self._save_dict = False
        self.dict_path = dict_path

    def __call__(self, **kwargs):
        """Report updated training status.
        Pass in `done=True` when the training job is completed.

        Args:
            kwargs: Latest training result status.
        Example
        -------
        >>> reporter(accuracy=1, training_iters=4)
        """
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
