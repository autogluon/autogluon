from typing import NamedTuple, List, Dict
import time
import logging
import numpy as np

logger = logging.getLogger(__name__)


class ProfilingBlock(NamedTuple):
    meta: dict
    time_stamp: float
    durations: Dict[str, List[float]]


class SimpleProfiler(object):
    """
    Useful to profile time of recurring computations, for example
    `get_config` calls in searchers.

    Measurements are divided into blocks. A block is started by `begin_block`.
    Each block stores meta data, a time stamp when `begin_block` was called
    (relative to the time stamp for the first block, which is 0), and a dict of
    lists of durations, whose keys are tags. A tag corresponds to a range of
    code to be profiled. It may be executed many times within a block,
    therefore lists of durations.

    Tags can have multiple levels of prefixes, corresponding to brackets.

    """

    def __init__(self):
        self.records = list()
        self.start_time = dict()
        self.time_stamp_first_block = None
        self.meta_keys = None
        self.prefix = ''

    def begin_block(self, meta: dict):
        assert not self.start_time, \
            "Timers for these tags still running:\n{}".format(
                self.start_time.keys())
        meta_keys = tuple(sorted(meta.keys()))
        if self.time_stamp_first_block is None:
            self.meta_keys = meta_keys
            self.time_stamp_first_block = time.time()
        else:
            assert meta_keys == self.meta_keys, \
                "meta.keys() = {}, but must be the same as for all previous meta dicts ({})".format(
                    meta_keys, self.meta_keys)
        time_stamp = time.time() - self.time_stamp_first_block
        new_block = ProfilingBlock(
            meta=meta.copy(),
            time_stamp=time_stamp,
            durations=dict())
        self.records.append(new_block)
        self.prefix = ''

    def push_prefix(self, prefix: str):
        assert '_' not in prefix, "Prefix must not contain '_'"
        self.prefix += (prefix + '_')

    def pop_prefix(self):
        lpref = len(self.prefix)
        assert lpref > 0, "Prefix is empty"
        pos = self.prefix.rfind('_', 0, lpref - 1)
        if pos != -1:
            self.prefix = self.prefix[:(pos + 1)]
        else:
            self.prefix = ''

    def start(self, tag: str):
        assert self.records, \
            "No block has been started yet (use 'begin_block')"
        tag = self.prefix + tag
        assert tag not in self.start_time, \
            "Timer for '{}' already running".format(tag)
        self.start_time[tag] = time.time()

    def stop(self, tag: str):
        assert self.records, \
            "No block has been started yet (use 'begin_block')"
        tag = self.prefix + tag
        assert tag in self.start_time, \
            "Timer for '{}' does not exist".format(tag)
        duration = time.time() - self.start_time[tag]
        block = self.records[-1]
        if tag in block.durations:
            block.durations[tag].append(duration)
        else:
            block.durations[tag] = [duration]
        del self.start_time[tag]

    def clear(self):
        remaining_tags = list(self.start_time.keys())
        if remaining_tags:
            logger.warning("Timers for these tags not stopped (will be removed):\n{}".format(
                remaining_tags))
        self.start_time = dict()

    def records_as_dict(self) -> dict:
        """
        Return records as a dict of lists, can be converted into Pandas
        data-frame by:

            pandas.DataFrame.fromDict(...)

        Each entry corresponds to a column.
        """
        if len(self.records) == 0:
            return dict()
        # For each tag, we emit the following columns: tag_num, tag_mean,
        # tag_std
        data = {'time_stamp': []}
        union_tags = self._union_of_tags()
        suffices = ('_num', '_mean', '_std')
        for tag in union_tags:
            for suffix in suffices:
                data[tag + suffix] = []
        for k in self.meta_keys:
            data[k] = []
        # fill the columns row by row
        for block in self.records:
            data['time_stamp'].append(block.time_stamp)
            for k, v in block.meta.items():
                data[k].append(v)
            for tag in union_tags:
                for suffix in suffices:
                    data[tag + suffix].append(0)
            for tag, durations in block.durations.items():
                data[tag + '_num'][-1] = len(durations)
                data[tag + '_mean'][-1] = np.mean(durations)
                data[tag + '_std'][-1] = np.std(durations)
        return data

    def _union_of_tags(self):
        union_tags = set()
        for block in self.records:
            union_tags.update(block.durations.keys())
        return union_tags
