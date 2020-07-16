# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: disable=ungrouped-imports
"""DatasetLoader. An extension of Gluon data loader that allows
reading and processing multiple files on-the-fly.
"""

__all__ = ['DatasetLoader']

import io
import os
import glob
import pickle
import warnings
import multiprocessing
from functools import partial

import numpy as np
from mxnet import context
from mxnet.gluon.data import ArrayDataset, SimpleDataset
from mxnet.gluon.data.dataloader import ForkingPickler, _as_in_context, \
    default_mp_batchify_fn, default_batchify_fn

# manager for creating shared object
_manager = None
_dataset = None


def _initialize_dataset_worker(manager):
    global _manager
    _manager = manager


class NumpyDataset(ArrayDataset):
    """A dataset wrapping over a Numpy binary (.npy, .npz) file.

    If the file is a .npy file, then a single numpy array is loaded.
    If the file is a .npz file with multiple arrays, then a list of
    numpy arrays are loaded, ordered by their key in the archive.

    Sparse matrix is not yet supported.

    Parameters
    ----------
    filename : str
        Path to the .npy or .npz file.
    kwargs
        Keyword arguments are passed to np.load.

    Properties
    ----------
    keys: list of str or None
        The list of keys loaded from the .npz file.
    """

    def __init__(self, filename, **kwargs):
        arrs = np.load(filename, **kwargs)
        keys = None
        data = []
        if filename.endswith('.npy'):
            data.append(arrs)
        elif filename.endswith('.npz'):
            keys = sorted(arrs.keys())
            for key in keys:
                data.append(arrs[key])
        else:
            raise ValueError('Unsupported extension: %s' % filename)
        self._keys = keys
        super(NumpyDataset, self).__init__(*data)

    @property
    def keys(self):
        return self._keys

    def get_field(self, field):
        """Return the dataset corresponds to the provided key.

        Example::
            a = np.ones((2,2))
            b = np.zeros((2,2))
            np.savez('data.npz', a=a, b=b)
            dataset = NumpyDataset('data.npz')
            data_a = dataset.get_field('a')
            data_b = dataset.get_field('b')

        Parameters
        ----------
        field : str
            The name of the field to retrieve.
        """
        idx = self._keys.index(field)
        return self._data[idx]


class _PathDataset(SimpleDataset):
    """A simple Datasets containing a list of paths given the file_pattern.

    Parameters
    ----------
    file_pattern: str
        Path to the input text files.
    """

    def __init__(self, file_pattern):
        if not isinstance(file_pattern, str):
            raise TypeError('file_pattern must be str, but got %s' % type(file_pattern))
        files = []
        for pattern in file_pattern.split(','):
            files.extend(glob.glob(os.path.expanduser(pattern.strip())))
        files = sorted(files)
        if len(files) == 0:
            raise ValueError('Cannot find any file with path "%s"' % file_pattern)
        super(_PathDataset, self).__init__(files)


def _dataset_worker_fn(urls, dataset_fn, batch_sampler_fn):
    """Function to generate datasets and batch sampler for each worker."""
    global _manager, _dataset
    dataset = dataset_fn(urls)
    batch_sampler = batch_sampler_fn(dataset)
    if _manager:
        dataset = _manager.list(zip(*dataset._data))
    _dataset = dataset
    return dataset, batch_sampler


def _batch_worker_fn(samples, batchify_fn, dataset=None, counter=None):
    """Function for processing data in worker process."""
    # pylint: disable=unused-argument
    # it is required that each worker process has to fork a new MXIndexedRecordIO handle
    # preserving dataset as global variable can save tons of overhead and is safe in new process
    if len(dataset[0]) > 1:
        if isinstance(samples[0], (list, tuple)):
            batch = [batchify_fn([dataset[i] for i in shard]) for shard in samples]
        else:
            batch = batchify_fn([dataset[i] for i in samples])
    else:
        if isinstance(samples[0], (list, tuple)):
            batch = [batchify_fn([dataset[i][0] for i in shard]) for shard in samples]
        else:
            batch = batchify_fn([dataset[i][0] for i in samples])
    buf = io.BytesIO()
    ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(batch)
    return buf.getvalue(), counter


class _MultiBatchWorkerIter:
    """Internal multi-worker iterator for DataLoader."""

    def __init__(self, worker_pool, batchify_fn, dataset_iter=None,
                 pin_memory=False, worker_fn=_batch_worker_fn, prefetch=0,
                 manager=None):
        self._worker_pool = worker_pool
        self._batchify_fn = batchify_fn
        self._data_buffer = {}
        self._rcvd_idx = 0
        self._sent_idx = 0
        self._dataset_iter = iter(dataset_iter)
        self._worker_fn = worker_fn
        self._pin_memory = pin_memory
        self._prefetch = prefetch
        self._dataset = None
        self._batch_iter = None
        self._manager = manager

        # datasets reference list
        self._dataset_refs = []

        # counter reference dict
        self._counter_ref = {}

        # pre-fetch
        for _ in range(self._prefetch):
            self._push_next()

    def _count_dataset_ref(self, new_dataset):
        dataset_refs = []
        for dataset in self._dataset_refs:
            if self._counter_ref[id(dataset)].value > 0:
                dataset_refs.append(dataset)
            else:
                del self._counter_ref[id(dataset)]
        if self._dataset:
            if self._counter_ref[id(self._dataset)].value > 0:
                if id(new_dataset) != id(self._dataset):
                    dataset_refs.append(self._dataset)
            else:
                del self._counter_ref[id(self._dataset)]

        self._dataset_refs = dataset_refs

    def _next_dataset(self):
        try:
            dataset, batch_sampler = next(self._dataset_iter)
        except StopIteration:
            return None
        return dataset, batch_sampler

    def _push_next(self):
        """Assign next batch workload to workers."""
        if self._batch_iter is not None:
            r = next(self._batch_iter, None)
        else:
            r = None
        if r is None:
            result = self._next_dataset()
            if result is None:
                return
            else:
                dataset, batch_sampler = result
                # Without checking the reference counts of previous datasets in the master process,
                # the key error can be triggered occasionally. This may be a bug in Python.
                self._count_dataset_ref(dataset)
                self._dataset = dataset
                # initialize reference counter
                if id(dataset) not in self._counter_ref:
                    self._counter_ref[id(dataset)] = self._manager.Value('i', 0)
                self._batch_iter = iter(batch_sampler)
                self._push_next()
        else:
            counter = self._counter_ref[id(self._dataset)]
            counter.value += 1
            async_ret = self._worker_pool.apply_async(
                self._worker_fn, (r, self._batchify_fn, self._dataset, counter))
            self._data_buffer[self._sent_idx] = async_ret
            self._sent_idx += 1

    def __next__(self):
        self._push_next()
        if self._rcvd_idx == self._sent_idx:
            assert not self._data_buffer, 'Data buffer should be empty at this moment'
            raise StopIteration

        assert self._rcvd_idx < self._sent_idx, 'rcvd_idx must be smaller than sent_idx'
        assert self._rcvd_idx in self._data_buffer, 'fatal error with _push_next, rcvd_idx missing'
        ret = self._data_buffer.pop(self._rcvd_idx)
        batch, counter = ret.get()
        batch = pickle.loads(batch)
        counter.value -= 1
        if self._pin_memory:
            batch = _as_in_context(batch, context.cpu_pinned())
        self._rcvd_idx += 1
        return batch

    def next(self):
        return self.__next__()

    def __iter__(self):
        return self


class _MultiDatasetWorkerIter:
    """Internal multi-worker iterator for DataLoader."""

    def __init__(self, worker_pool, file_sampler,
                 dataset_fn, batch_sampler_fn,
                 worker_fn=_dataset_worker_fn,
                 prefetch=0, dataset=None, circle_length=1,
                 cached=False, num_max_cached=0):
        if cached:
            assert num_max_cached > 0, \
                'When cached is turned on, num_max_cached must be positive.'
        self._worker_pool = worker_pool
        self._dataset_fn = dataset_fn
        self._batch_sampler_fn = batch_sampler_fn
        self._worker_fn = worker_fn
        self._prefetch = prefetch
        self._circle_length = circle_length
        self._cached = cached
        self._num_max_cached = num_max_cached

        # send and receive index for datasets
        self._rcvd_idx = 0
        self._sent_idx = 0
        self._data_buffer = {}

        self._dataset = [dataset[i] for i in iter(file_sampler)]
        self._num_datasets = len(self._dataset)

        # construct cached list
        self._cached_dataset = []

        # pre-fetch
        for _ in range(self._prefetch):
            self._push_next_dataset()

    def _push_next_dataset(self):
        """Assign next dataset workload to workers."""
        current_dataset_idx = self._sent_idx * self._circle_length
        if current_dataset_idx < self._num_datasets:
            circle_length = min(self._circle_length,
                                self._num_datasets - current_dataset_idx)
            urls = [self._dataset[current_dataset_idx + i] for i in range(circle_length)]
        else:
            return
        # push to worker asynchronously
        async_ret = self._worker_pool.apply_async(
            self._worker_fn, (urls, self._dataset_fn, self._batch_sampler_fn))
        # data buffer stores the async result
        self._data_buffer[self._sent_idx] = async_ret
        self._sent_idx += 1

    def _next_dataset(self):
        """Retrieve the next dataset. Returns None if no dataset is available."""
        if self._rcvd_idx == self._sent_idx:
            assert not self._data_buffer, 'Data buffer should be empty at this moment'
            return None

        assert self._rcvd_idx < self._sent_idx, \
            'rcvd_idx must be smaller than sent_idx'
        assert self._rcvd_idx in self._data_buffer, \
            'fatal error with _next_dataset, rcvd_idx missing'

        if len(self._cached_dataset) == 0 or self._data_buffer[self._rcvd_idx].ready():
            ret = self._data_buffer.pop(self._rcvd_idx)
            dataset, batch_sampler = ret.get()
            self._rcvd_idx += 1
            if self._cached and len(self._cached_dataset) < self._num_max_cached:
                self._cached_dataset.append((dataset, batch_sampler))
        else:
            dataset, batch_sampler = self._cached_dataset.pop(0)

        return dataset, batch_sampler

    def __next__(self):
        """Next dataset"""
        self._push_next_dataset()
        result = self._next_dataset()

        if result is None:
            raise StopIteration

        return result

    def next(self):
        """Next dataset"""
        return self.__next__()

    def __iter__(self):
        """Returns the iterator object"""
        return self


class DatasetLoader:
    """Loads data from a list of datasets and returns mini-batches of data.

    One dataset is loaded at a time.

    Parameters
    ----------
    file_patterns: str
        Path to the input text files.
    file_sampler : str or gluon.data.Sampler, defaults to 'random'
        The sampler used to sample a file. The following string values are supported:

        - 'sequential': SequentialSampler
        - 'random': RandomSampler
    dataset_fn : DatasetFn, callable
        Callable object to generate a gluon.data.Dataset given a url.
    batch_sampler_fn : SamplerFn, callable
        Callable object to generate a gluon.data.sampler.Sampler given a dataset.
    dataset_params : dict, default is None
        Dictionary of parameters passed to dataset_fn.
    batch_sampler_params : dict, default is None
        Dictionary of parameters passed to batch_sampler_fn.
    batchify_fn : callable
        Callback function to allow users to specify how to merge samples
        into a batch. Defaults to `default_batchify_fn`::

            def default_batchify_fn(data):
                if isinstance(data[0], nd.NDArray):
                    return nd.stack(*data)
                elif isinstance(data[0], tuple):
                    data = zip(*data)
                    return [default_batchify_fn(i) for i in data]
                else:
                    data = np.asarray(data)
                    return nd.array(data, dtype=data.dtype)
    num_dataset_workers : int
        Number of worker process for dataset creation.
    num_batch_workers : int
        Number of worker process for batch creation.
    pin_memory : boolean, default False
        If ``True``, the dataloader will copy NDArrays into pinned memory
        before returning them. Copying from CPU pinned memory to GPU is faster
        than from normal CPU memory. At the same time, it increases GPU memory.
    circle_length : int, default is 1
        The number of files to be read at the same time. When `circle_length` is larger than 1,
        we merge `circle_length` number of files.
    dataset_prefetch : int, default is `num_dataset_workers`
        The number of prefetching datasets only works if `num_workers` > 0.
        If `prefetch` > 0, it allow worker process to prefetch certain datasets before
        acquiring data from iterators.
        Note that using large prefetching batch will provide smoother bootstrapping performance,
        but will consume more memory. Using smaller number may forfeit the purpose of using
        multiple worker processes, try reduce `num_dataset_workers` in this case.
        By default it defaults to `num_dataset_workers`.
    batch_prefetch : int, default is `num_batch_workers * 2`
        The number of prefetching batches only works if `num_workers` > 0.
        If `prefetch` > 0, it allow worker process to prefetch certain batches before
        acquiring data from iterators.
        Note that using large prefetching batch will provide smoother bootstrapping performance,
        but will consume more shared_memory. Using smaller number may forfeit the purpose of using
        multiple worker processes, try reduce `num_batch_workers` in this case.
        By default it defaults to `num_batch_workers * 2`.
    dataset_cached : bool, default is False
        Whether or not to cache last processed dataset. Each processed dataset can
        only be cached for once. When there is no new available processed dataset to be fetched,
        we pop a cached processed dataset.
    num_max_dataset_cached : int, default is 0
        Maximum number of cached datasets. It is valid only if `dataset_cached` is True
    """

    def __init__(self, file_patterns, file_sampler,
                 dataset_fn=None, batch_sampler_fn=None,
                 dataset_params=None, batch_sampler_params=None, batchify_fn=None,
                 num_dataset_workers=0, num_batch_workers=0,
                 pin_memory=False, circle_length=1,
                 dataset_prefetch=None, batch_prefetch=None,
                 dataset_cached=False, num_max_dataset_cached=0):
        assert num_dataset_workers >= 0, \
            'num_dataset_workers must be non-negative'
        assert num_batch_workers >= 0, \
            'num_batch_workers must be non-negative'
        if num_batch_workers > 0:
            assert num_dataset_workers > 0, \
                'num_dataset_workers must be positive when num_batch_workers > 0'
        else:
            if num_dataset_workers > 0:
                warnings.warn('The multi-processing functionalities for both dataset and'
                              ' batch sampling are disabled when num_batch_workers=0 though '
                              'num_dataset_workers={} > 0'.format(num_dataset_workers))
        assert circle_length >= 1, \
            'circle_length must be larger than or equal to 1'
        if dataset_cached:
            assert num_max_dataset_cached > 0, \
                'When dataset_cached is True, num_max_dataset_cached must be positive'

        self._dataset = _PathDataset(file_patterns)
        self._file_sampler = file_sampler

        assert dataset_fn is not None, 'dataset_fn is not given.'
        assert batch_sampler_fn is not None, 'batch_sampler_fn is not given.'
        if dataset_params is not None:
            self._dataset_fn = partial(dataset_fn, **dataset_params)
        else:
            self._dataset_fn = dataset_fn
        if batch_sampler_params is not None:
            self._batch_sampler_fn = partial(batch_sampler_fn, **batch_sampler_params)
        else:
            self._batch_sampler_fn = batch_sampler_fn

        self._num_dataset_workers = num_dataset_workers
        self._num_batch_workers = num_batch_workers
        self._dataset_prefetch = max(0, int(dataset_prefetch)
                                     if dataset_prefetch is not None else self._num_dataset_workers)
        self._batch_prefetch = max(0, int(batch_prefetch)
                                   if batch_prefetch is not None else 2 * self._num_batch_workers)

        self._pin_memory = pin_memory
        self._circle_length = circle_length
        self._dataset_cached = dataset_cached
        self._num_max_dataset_cached = num_max_dataset_cached

        self._manager = None
        self._dataset_worker_pool = None
        if self._num_dataset_workers > 0:
            self._manager = multiprocessing.Manager()
            self._dataset_worker_pool = multiprocessing.Pool(self._num_dataset_workers,
                                                             initializer=_initialize_dataset_worker,
                                                             initargs=[self._manager])
        self._batch_worker_pool = None
        if self._num_batch_workers > 0:
            self._batch_worker_pool = multiprocessing.Pool(self._num_batch_workers)
        if batchify_fn is None:
            if self._num_batch_workers > 0:
                self._batchify_fn = default_mp_batchify_fn
            else:
                self._batchify_fn = default_batchify_fn
        else:
            self._batchify_fn = batchify_fn

    def __iter__(self):
        if self._num_dataset_workers == 0:
            def _same_process_iter():
                urls = []
                dataset = [self._dataset[i] for i in iter(self._file_sampler)]
                for i, url in enumerate(dataset):
                    urls.append(url)
                    if i < len(dataset) - 1:
                        if len(urls) < self._circle_length:
                            continue
                    if self._circle_length == 1:
                        urls = urls[0]
                    dataset, batch_sampler = _dataset_worker_fn(urls, self._dataset_fn,
                                                                self._batch_sampler_fn)
                    for batch in batch_sampler:
                        ret = self._batchify_fn([dataset[idx] for idx in batch])
                        if self._pin_memory:
                            ret = _as_in_context(ret, context.cpu_pinned())
                        yield ret
                    urls = []

            return _same_process_iter()

        # multi-worker
        dataset_iter = _MultiDatasetWorkerIter(self._dataset_worker_pool,
                                               worker_fn=_dataset_worker_fn,
                                               dataset=self._dataset,
                                               file_sampler=self._file_sampler,
                                               dataset_fn=self._dataset_fn,
                                               batch_sampler_fn=self._batch_sampler_fn,
                                               prefetch=self._dataset_prefetch,
                                               circle_length=self._circle_length,
                                               cached=self._dataset_cached,
                                               num_max_cached=self._num_max_dataset_cached)
        return _MultiBatchWorkerIter(self._batch_worker_pool, self._batchify_fn, dataset_iter,
                                     pin_memory=self._pin_memory, worker_fn=_batch_worker_fn,
                                     prefetch=self._batch_prefetch, manager=self._manager)

    def __del__(self):
        if self._dataset_worker_pool:
            # manually terminate due to a bug that pool is not automatically terminated
            # https://bugs.python.org/issue34172
            assert isinstance(self._dataset_worker_pool, multiprocessing.pool.Pool)
            self._dataset_worker_pool.terminate()
        if self._batch_worker_pool:
            assert isinstance(self._batch_worker_pool, multiprocessing.pool.Pool)
            self._batch_worker_pool.terminate()
