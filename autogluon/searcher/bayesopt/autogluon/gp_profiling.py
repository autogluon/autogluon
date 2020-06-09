from typing import NamedTuple
import time
import logging

from autogluon.searcher.bayesopt.datatypes.tuning_job_state import \
    TuningJobState

logger = logging.getLogger(__name__)


class ProfilingData(NamedTuple):
    id: int
    tag: str
    duration: float
    num_labeled: int
    num_pending: int
    fit_hyperparams: bool

    def to_tuple(self):
        return (self.id,
                self.tag,
                self.duration,
                self.num_labeled,
                self.num_pending,
                self.fit_hyperparams)

    @staticmethod
    def field_names():
        return ('id',
                'tag',
                'duration',
                'num_labeled',
                'num_pending',
                'fit_hyperparams')


class GPMXNetSimpleProfiler(object):
    def __init__(self):
        self.records = list()
        self.block = None
        self.start_time = dict()
        self.id_counter = 0

    def set_state(self, state: TuningJobState,
                  fit_hyperparams: bool):
        assert not self.start_time, \
            "Timers for these tags still running:\n{}".format(
                self.start_time.keys())
        self.block = ProfilingData(
            id=self.id_counter, tag='', duration=0.,
            num_labeled=len(state.candidate_evaluations),
            num_pending=len(state.pending_evaluations),
            fit_hyperparams=fit_hyperparams
        )
        self.id_counter += 1

    def start(self, tag: str):
        assert self.block is not None
        assert tag not in self.start_time, \
            "Timer for '{}' already running".format(tag)
        self.start_time[tag] = time.process_time()

    def stop(self, tag: str):
        assert tag in self.start_time, \
            "Timer for '{}' does not exist".format(tag)
        duration = time.process_time() - self.start_time[tag]
        self.records.append(
            self.block._replace(duration=duration, tag=tag))
        del self.start_time[tag]

    def clear(self):
        remaining_tags = list(self.start_time.keys())
        if remaining_tags:
            logger.warning("Timers for these tags not stopped (will be removed):\n{}".format(
                remaining_tags))
        self.start_time = dict()
