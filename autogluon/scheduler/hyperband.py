import os
import pickle
import logging
import threading
import numpy as np
import multiprocessing as mp

from .scheduler import *
from .fifo import FIFO_Scheduler
from .resource_manager import Resources
from .reporter import StatusReporter

__all__ = ['Hyperband_Scheduler']

logger = logging.getLogger(__name__)

# Async version of Hyperband used in computation heavy tasks such as deep learning
class Hyperband_Scheduler(FIFO_Scheduler):
    """Implements the Async Hyperband
    This should provide similar theoretical performance as HyperBand but
    avoid straggler issues that HyperBand faces. One implementation detail
    is when using multiple brackets, task allocation to bracket is done
    randomly with over a softmax probability.
    See https://arxiv.org/abs/1810.05934
    Args:
        train_fn (callable): A task launch function for training.
            Note: please add the `@autogluon_method` decorater to the original function.
        args (object): Default arguments for launching train_fn.
        resource (dict): Computation resources.
            For example, `{'num_cpus':2, 'num_gpus':1}`
        searcher (object): Autogluon searcher.
            For example, autogluon.searcher.RandomSampling
        time_attr (str): A training result attr to use for comparing time.
            Note that you can pass in something non-temporal such as
            `training_epoch` as a measure of progress, the only requirement
            is that the attribute should increase monotonically.
        reward_attr (str): The training result objective value attribute. As
            with `time_attr`, this may refer to any objective value. Stopping
            procedures will use this attribute.
        max_t (float): max time units per task. Trials will be stopped after
            max_t time units (determined by time_attr) have passed.
        grace_period (float): Only stop tasks at least this old in time.
            The units are the same as the attribute named by `time_attr`.
        reduction_factor (float): Used to set halving rate and amount. This
            is simply a unit-less scalar.
        brackets (int): Number of brackets. Each bracket has a different
            halving rate, specified by the reduction factor.
    """
    def __init__(self, train_fn, args, resource, searcher,
                 checkpoint=None, resume=False,
                 num_trials=None,
                 time_attr="training_epoch",
                 reward_attr="accuracy",
                 max_t=100, grace_period=10,
                 reduction_factor=4, brackets=1):
        super(Hyperband_Scheduler, self).__init__(train_fn, args, resource, searcher,
                                                  checkpoint, resume, num_trials,
                                                  time_attr, reward_attr)
        self.terminator = Hyperband_Manager(time_attr, reward_attr, max_t, grace_period,
                                            reduction_factor, brackets)

    def add_task(self, task):
        # adding the task
        logger.debug("Adding A New Task {}".format(task))
        Hyperband_Scheduler.RESOURCE_MANAGER._request(task.resources)
        with self.LOCK:
            reporter = StatusReporter()
            task.args['reporter'] = reporter
            self.terminator.on_task_add(task)
            # main process
            tp = mp.Process(target=Hyperband_Scheduler._run_task, args=(
                            task.fn, task.args, task.resources,
                            Hyperband_Scheduler.RESOURCE_MANAGER))
            # reporter thread
            checkpoint_semaphore = mp.Semaphore(0) if self._checkpoint else None
            rp = threading.Thread(target=self._run_reporter,
                                  args=(task, tp, reporter, self.searcher, self.terminator,
                                        checkpoint_semaphore))
            tp.start()
            rp.start()
            # checkpoint thread
            if self._checkpoint is not None:
                sp = threading.Thread(target=self._run_checkpoint, args=(checkpoint_semaphore,))
                sp.start()
            self.SCHEDULED_TASKS.append({'TASK_ID': task.task_id, 'Config': task.args['config'],
                                         'Process': tp, 'ReporterProcess': rp})

    def _run_reporter(self, task, task_process, reporter, searcher, terminator,
                      checkpoint_semaphore):
        last_result = None
        while task_process.is_alive():
            reported_result = reporter.fetch()
            if 'done' in reported_result and reported_result['done'] is True:
                terminator.on_task_complete(task, last_result)
                task_process.join()
                if checkpoint_semaphore is not None:
                    checkpoint_semaphore.release()
                break
            self.add_training_result(task.task_id, reported_result[self._reward_attr])
            if terminator.on_task_report(task, reported_result):
                reporter.move_on()
            else:
                logger.debug('Removing task {} due to low performance'.format(task))
                last_result = reported_result
                last_result['terminated'] = True
                task_process.terminate()
                terminator.on_task_remove(task)
                task_process.join()
                Hyperband_Scheduler.RESOURCE_MANAGER._release(task.resources)
                if checkpoint_semaphore is not None:
                    checkpoint_semaphore.release()
                break
            last_result = reported_result
        searcher.update(task.args['config'], last_result[self._reward_attr])

    def state_dict(self, destination=None):
        destination = super(Hyperband_Scheduler, self).state_dict(destination)
        destination['terminator'] = pickle.dumps(self.terminator)
        return destination

    def load_state_dict(self, state_dict):
        super(Hyperband_Scheduler, self).load_state_dict(state_dict)
        self.terminator = pickle.loads(state_dict['terminator'])
        logger.info('Loading Terminator State {}'.format(self.terminator))

class Hyperband_Manager(object):
    """Hyperband Manager
    Args:
        time_attr (str): A training result attr to use for comparing time.
            Note that you can pass in something non-temporal such as
            `training_epoch` as a measure of progress, the only requirement
            is that the attribute should increase monotonically.
        reward_attr (str): The training result objective value attribute. As
            with `time_attr`, this may refer to any objective value. Stopping
            procedures will use this attribute.
        max_t (float): max time units per task. Trials will be stopped after
            max_t time units (determined by time_attr) have passed.
        grace_period (float): Only stop tasks at least this old in time.
            The units are the same as the attribute named by `time_attr`.
        reduction_factor (float): Used to set halving rate and amount. This
            is simply a unit-less scalar.
        brackets (int): Number of brackets. Each bracket has a different
            halving rate, specified by the reduction factor.
    """
    LOCK = mp.Lock()
    def __init__(self, time_attr='training_epoch',
                 reward_attr='accuracy',
                 max_t=100, grace_period=10,
                 reduction_factor=4, brackets=1):
        # attr
        self._reward_attr = reward_attr
        self._time_attr = time_attr
        # hyperband params
        self._task_info = {}  # Stores Task -> Bracket
        self._reduction_factor = reduction_factor
        self._max_t = max_t
        self._num_stopped = 0
        # Tracks state for new task add
        self._brackets = [
            _Bracket(grace_period, max_t, reduction_factor, s)
            for s in range(brackets)
        ]

    def on_task_add(self, task):
        sizes = np.array([len(b._rungs) for b in self._brackets])
        probs = np.e**(sizes - sizes.max())
        normalized = probs / probs.sum()
        idx = np.random.choice(len(self._brackets), p=normalized)
        with Hyperband_Manager.LOCK:
            self._task_info[task.task_id] = self._brackets[idx]

    def on_task_report(self, task, result):
        with Hyperband_Manager.LOCK:
            action = True
            if result[self._time_attr] >= self._max_t:
                action = False
            else:
                bracket = self._task_info[task.task_id]
                action = bracket.on_result(task, result[self._time_attr],
                                           result[self._reward_attr])
            if action == False:
                self._num_stopped += 1
            return action

    def on_task_complete(self, task, result):
        with Hyperband_Manager.LOCK:
            bracket = self._task_info[task.task_id]
            bracket.on_result(task, result[self._time_attr],
                              result[self._reward_attr])
            del self._task_info[task.task_id]

    def on_task_remove(self, task):
        with Hyperband_Manager.LOCK:
            del self._task_info[task.task_id]
 
    def __repr__(self):
        reprstr = self.__class__.__name__ + '(' +  \
            'reward_attr: ' + self._reward_attr + \
            ', time_attr: ' + self._time_attr + \
            ', reduction_factor: ' + str(self._reduction_factor) + \
            ', max_t: ' + str(self._max_t) + \
            ', brackets: ' + str(self._brackets) + \
             ')'
        return reprstr

# adapted from ray-project
class _Bracket():
    """Bookkeeping system to track the cutoffs.
    Rungs are created in reversed order so that we can more easily find
    the correct rung corresponding to the current iteration of the result.
    Example:
        >>> b = _Bracket(1, 10, 2, 3)
        >>> b.on_result(task1, 1, 2)  # CONTINUE
        >>> b.on_result(task2, 1, 4)  # CONTINUE
        >>> b.cutoff(b._rungs[-1][1]) == 3.0  # rungs are reversed
        >>> b.on_result(task3, 1, 1)  # STOP
        >>> b.cutoff(b._rungs[0][1]) == 2.0
    """

    def __init__(self, min_t, max_t, reduction_factor, s):
        self.rf = reduction_factor
        MAX_RUNGS = int(np.log(max_t / min_t) / np.log(self.rf) - s + 1)
        self._rungs = [(min_t * self.rf**(k + s), {})
                       for k in reversed(range(MAX_RUNGS))]

    def cutoff(self, recorded):
        if not recorded:
            return None
        return np.percentile(list(recorded.values()), (1 - 1 / self.rf) * 100)

    def on_result(self, task, cur_iter, cur_rew):
        action = True
        for milestone, recorded in self._rungs:
            if cur_iter < milestone or task.task_id in recorded:
                continue
            else:
                cutoff = self.cutoff(recorded)
                if cutoff is not None and cur_rew < cutoff:
                    action = False
                if cur_rew is None:
                    logger.warning("Reward attribute is None! Consider"
                                   " reporting using a different field.")
                else:
                    recorded[task.task_id] = cur_rew
                break
        return action

    def __repr__(self):
        iters = " | ".join([
            "Iter {:.3f}: {}".format(milestone, self.cutoff(recorded))
            for milestone, recorded in self._rungs
        ])
        return "Bracket: " + iters
