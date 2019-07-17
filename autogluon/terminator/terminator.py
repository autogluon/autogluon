import multiprocessing as mp
import numpy as np
import logging
from abc import ABCMeta, abstractmethod
from collections import defaultdict

__all__ = ['BaseTerminator', 'MedianStopping', 'HyperBand']
logger = logging.getLogger(__name__)


class BaseTerminator(metaclass=ABCMeta):

    def __init__(self, time_attr='epoch',
                 reward_attr='accuracy', *args, **kwargs):
        self._time_attr = time_attr
        self._reward_attr = reward_attr

    @abstractmethod
    def on_task_add(self, task_id):
        pass

    @abstractmethod
    def on_task_report(self, task_id, result):
        """
        :param task_id:
        :param result:
        :return: False if suggesting to terminate else True
        """
        pass

    @abstractmethod
    def on_task_complete(self, task_id, result):
        pass

    @abstractmethod
    def on_task_remove(self, task_id):
        pass

    @abstractmethod
    def __repr__(self):
        pass


#adopted from ray-project

class MedianStopping(BaseTerminator):

    def __init__(self, time_attr='epoch',
                 reward_attr='accuracy',
                 min_samples:int = 3,
                 abort_trial = True,
                 *args, **kwargs):
        super(HyperBand, self).__init__(time_attr, reward_attr, *args, **kwargs)
        self._min_samples = min_samples
        self._abort_trial = abort_trial
        self._completed_tasks = set()
        self._stopped_tasks = set()
        self._scores = defaultdict(list)

    def on_task_add(self, task_id):
        pass

    def on_task_report(self, task_id, result):

        self._scores.append(result[self._reward_attr])

        # don't evaluate if we don't have a min number of samples
        if len(self._scores) < self._min_samples:
            return True
        else:
            try:
                num_epoch = result['epoch']
            except KeyError:
                # consider all epochs so far
                num_epoch = -1

            self._evaluate_trial(task_id, result[self._reward_attr], num_epoch)

    def on_task_complete(self, task_id, result):
        self._scores.append(result[self._reward_attr])
        self._completed_tasks.add(task_id)

    def __repr__(self):
        median_score = self._get_median()
        task_scores = ''
        for t_id, scores in self._scores.items():
            task_scores += "{}:{}".format(t_id, scores)

        return 'task_scores:{}, median_score:{}'.format(task_scores, median_score)

    def _get_median(self, num_epoch=-1):
        if len(self._scores) < self._min_samples:
            return float("-inf")
        score_so_far = []

        for each_task_id, scores in self._scores.items():
            # where you should use mean or the best so far should be configurable
            score_so_far.append(np.mean(scores[:num_epoch]))

        return np.median(score_so_far)

    def _evaluate_trial(self, task_id, reward_attr, num_epoch=-1):

        median_score = self._get_median(num_epoch)
        task_best_score = np.max(self._scores[reward_attr])
        logger.info('task id:{}, task_best_score:{},'
                    'median score:{}, num_trials_so_far:{}'.format(task_id, task_best_score,
                                                                   median_score, len(self._scores)))
        if task_best_score < median_score:
            logger.info('Stopping task_id: {}'.format(task_id))
            self._stopped_tasks.add(task_id)
            return False
        else:
            return True


class HyperBand(BaseTerminator):
    """Hyperband Manager

    Args:
        time_attr (str): A training result attr to use for comparing time.
            Note that you can pass in something non-temporal such as
            `epoch` as a measure of progress, the only requirement
            is that the attribute should increase monotonically.
        reward_attr (str): The training result objective value attribute. As
            with `time_attr`, this may refer to any objective value. Stopping
            procedures will use this attribute.
        max_time (float): max time units per task. Trials will be stopped after
            max_t time units (determined by time_attr) have passed.
        grace_period (float): Only stop tasks at least this old in time.
            The units are the same as the attribute named by `time_attr`.
        reduction_factor (float): Used to set halving rate and amount. This
            is simply a unit-less scalar.
        brackets (int): Number of brackets. Each bracket has a different
            halving rate, specified by the reduction factor.
    """
    LOCK = mp.Lock()

    def __init__(self, time_attr='epoch',
                 reward_attr='accuracy',
                 max_time=100, grace_period=10,
                 reduction_factor=4, brackets=1,
                 *args, **kwargs):
        # attr
        super(HyperBand, self).__init__(time_attr, reward_attr, *args, **kwargs)
        # hyperband params
        self._task_info = {}  # Stores Task -> Bracket
        self._reduction_factor = reduction_factor
        self._max_time = max_time
        self._num_stopped = 0
        self._stopped_tasks = []
        # Tracks state for new task add
        self._brackets = [
            Bracket(grace_period, max_time, reduction_factor, s)
            for s in range(brackets)
        ]

    def on_task_add(self, task_id):
        sizes = np.array([len(b._rungs) for b in self._brackets])
        probs = np.e**(sizes - sizes.max())
        normalized = probs / probs.sum()
        idx = np.random.choice(len(self._brackets), p=normalized)
        self._task_info[task_id] = self._brackets[idx]

    def on_task_report(self, task_id, result):
        with HyperBand.LOCK:
            action = True
            if result[self._time_attr] >= self._max_t:
                action = False
            else:
                bracket = self._task_info[task_id.task_id]
                action = bracket.on_result(task_id, result[self._time_attr],
                                           result[self._reward_attr])
            if not action:
                self._num_stopped += 1
                self._stopped_tasks.append(task_id)
            return action

    def on_task_complete(self, task_id, result):
        with HyperBand.LOCK:
            bracket = self._task_info[task_id]
            bracket.on_result(task_id, result[self._time_attr],
                              result[self._reward_attr])
            del self._task_info[task_id]

    def on_task_remove(self, task_id):
        with HyperBand.LOCK:
            del self._task_info[task_id]

    def __repr__(self):
        reprstr = self.__class__.__name__ + '(' +  \
            'reward_attr: ' + self._reward_attr + \
            ', time_attr: ' + self._time_attr + \
            ', reduction_factor: ' + str(self._reduction_factor) + \
            ', max_t: ' + str(self._max_time) + \
            ', brackets: ' + str(self._brackets) + \
             ')'
        return reprstr


# adapted from the ray-project
class Bracket:
    """Bookkeeping system to track the cutoffs.
    Rungs are created in reversed order so that we can more easily find
    the correct rung corresponding to the current iteration of the result.
    Example:
        >>> b = Bracket(1, 10, 2, 3)
        >>> b.on_result(task1, 1, 2)  # CONTINUE
        >>> b.on_result(task2, 1, 4)  # CONTINUE
        >>> b.cutoff(b._rungs[-1][1]) == 3.0  # rungs are reversed
        >>> b.on_result(task3, 1, 1)  # STOP
        >>> b.cutoff(b._rungs[0][1]) == 2.0
    """

    def __init__(self, min_t, max_t, reduction_factor, s):
        self.rf = reduction_factor
        MAX_RUNGS = int(np.log(max_t / min_t) / np.log(self.rf) - s + 1)
        self._rungs = [(min_t * self.rf ** (k + s), {})
                       for k in reversed(range(MAX_RUNGS))]

    def cutoff(self, recorded):
        if not recorded:
            return None
        return np.percentile(list(recorded.values()), (1 - 1 / self.rf) * 100)

    def on_result(self, task_id, cur_iter, cur_rew):
        action = True
        for milestone, recorded in self._rungs:
            if cur_iter < milestone or task_id in recorded:
                continue
            else:
                cutoff = self.cutoff(recorded)
                if cutoff is not None and cur_rew < cutoff:
                    action = False
                if cur_rew is None:
                    logger.warning("Reward attribute is None! Consider"
                                   " reporting using a different field.")
                else:
                    recorded[task_id] = cur_rew
                break
        return action

    def __repr__(self):
        iters = " | ".join([
            "Iter {:.3f}: {}".format(milestone, self.cutoff(recorded))
            for milestone, recorded in self._rungs
        ])
        return "Bracket: " + iters