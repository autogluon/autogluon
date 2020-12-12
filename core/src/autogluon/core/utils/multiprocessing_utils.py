import logging
import multiprocessing
import threading
from contextlib import contextmanager

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AtomicCounter(object):
    def __init__(self, initial_value=0):
        self._counter = multiprocessing.Value('i', initial_value)
        self.lock = RWLock()

    def get_and_increment(self):
        with write_lock(self.lock):
            value = self._counter.value
            self._counter.value = value + 1
        return value

    def increment_and_get(self):
        with write_lock(self.lock):
            self._counter.value = self._counter.value + 1
            value = self._counter.value
        return value

    def get(self):
        with read_lock(self.lock):
            value = self._counter.value
        return value

    def set(self, value):
        with write_lock(self.lock):
            self._counter.value = value


class RWLock:
    """Synchronization object used in a solution of so-called second
    readers-writers problem. In this problem, many readers can simultaneously
    access a share, and a writer has an exclusive access to this share.
    Additionally, the following constraints should be met:
    1) no reader should be kept waiting if the share is currently opened for
        reading unless a writer is also waiting for the share,
    2) no writer should be kept waiting for the share longer than absolutely
        necessary.

    The implementation is based on [1, secs. 4.2.2, 4.2.6, 4.2.7]
    with a modification -- adding an additional lock (C{self.__readers_queue})
    -- in accordance with [2].

    By default multiprocessing locks are used.

    Sources:
    [1] A.B. Downey: "The little book of semaphores", Version 2.1.5, 2008
    [2] P.J. Courtois, F. Heymans, D.L. Parnas:
        "Concurrent Control with 'Readers' and 'Writers'",
        Communications of the ACM, 1971 (via [3])
    [3] http://en.wikipedia.org/wiki/Readers-writers_problem
    [4] https://code.activestate.com/recipes/577803-reader-writer-lock-with-priority-for-writers/
    """

    def __init__(self, use_multiprocess_locks=True):
        self.__read_switch = _LightSwitch()
        self.__write_switch = _LightSwitch()
        self.__no_readers = multiprocessing.Lock() if use_multiprocess_locks else threading.Lock
        self.__no_writers = multiprocessing.Lock() if use_multiprocess_locks else threading.Lock
        self.__readers_queue = multiprocessing.Lock() if use_multiprocess_locks else threading.Lock
        """A lock giving an even higher priority to the writer in certain
        cases (see [2] for a discussion)"""

    def reader_acquire(self):
        self.__readers_queue.acquire()
        self.__no_readers.acquire()
        self.__read_switch.acquire(self.__no_writers)
        self.__no_readers.release()
        self.__readers_queue.release()

    def reader_release(self):
        self.__read_switch.release(self.__no_writers)

    def writer_acquire(self):
        self.__write_switch.acquire(self.__no_readers)
        self.__no_writers.acquire()

    def writer_release(self):
        self.__no_writers.release()
        self.__write_switch.release(self.__no_readers)


class _LightSwitch:
    """An auxiliary "light switch"-like object. The first thread turns on the
    "switch", the last one turns it off (see [1, sec. 4.2.2] for details)."""

    def __init__(self):
        self.__counter = 0
        self.__mutex = threading.Lock()

    def acquire(self, lock):
        self.__mutex.acquire()
        self.__counter += 1
        if self.__counter == 1:
            lock.acquire()
        self.__mutex.release()

    def release(self, lock):
        self.__mutex.acquire()
        self.__counter -= 1
        if self.__counter == 0:
            lock.release()
        self.__mutex.release()


@contextmanager
def read_lock(lock: RWLock):
    lock.reader_acquire()
    yield
    lock.reader_release()


@contextmanager
def write_lock(lock: RWLock):
    lock.writer_acquire()
    yield
    lock.writer_release()


def dataframe_transform_parallel(
        df, transformer
):
    cpu_count = multiprocessing.cpu_count()
    workers_count = int(round(cpu_count))
    logger.log(15, 'Dataframe_transform_parallel running pool with ' + str(workers_count) + ' workers')
    df_chunks = np.array_split(df, workers_count)
    df_list = execute_multiprocessing(workers_count=workers_count, transformer=transformer, chunks=df_chunks)
    df_combined = pd.concat(df_list, axis=0, ignore_index=True)
    return df_combined


# If multiprocessing_method is 'fork', initialization time scales linearly with current allocated memory, dramatically slowing down runs. forkserver makes this time constant
def execute_multiprocessing(workers_count, transformer, chunks, multiprocessing_method='forkserver'):
    logger.log(15, 'Execute_multiprocessing starting worker pool...')
    ctx = multiprocessing.get_context(multiprocessing_method)
    with ctx.Pool(workers_count) as pool:
        out = pool.map(transformer, chunks)
    return out


def force_forkserver():
    """
    Forces forkserver multiprocessing mode if not set. This is needed for HPO and CUDA.
    The CUDA runtime does not support the fork start method: either the spawn or forkserver start method are required.
    forkserver is used because spawn is still affected by locking issues
    """
    if ('forkserver' in multiprocessing.get_all_start_methods()) & (not is_forkserver_enabled()):
        logger.warning('WARNING: changing multiprocessing start method to forkserver')
        multiprocessing.set_start_method('forkserver', force=True)


def is_forkserver_enabled():
    """
    Return True if current multiprocessing start method is forkserver.
    """
    return multiprocessing.get_start_method(allow_none=True) == 'forkserver'


def is_fork_enabled():
    """
    Return True if current multiprocessing start method is fork.
    """
    return multiprocessing.get_start_method(allow_none=True) == 'fork'
