import logging
import numpy as np
import multiprocessing as mp

logger = logging.getLogger(__name__)


class HyperbandStopping_Manager(object):
    """Hyperband Manager

    Implements stopping rule which uses the brackets and rung levels defined
    in Hyperband. The overall algorithm is NOT what is published as ASHA
    (see HyperbandPromotion_Manager), but rather something resembling the
    median rule.

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

    def __init__(
            self, time_attr, reward_attr, max_t, grace_period,
            reduction_factor, brackets):
        # attr
        self._reward_attr = reward_attr
        self._time_attr = time_attr
        # hyperband params
        self._task_info = dict()  # task_id -> bracket_id
        self._reduction_factor = reduction_factor
        self._max_t = max_t
        self._num_stopped = 0
        self._brackets = []
        for s in range(brackets):
            bracket = _Bracket(grace_period, max_t, reduction_factor, s)
            if not bracket._rungs:
                break
            self._brackets.append(bracket)

    def on_task_add(self, task, **kwargs):
        """
        Since the bracket has already been sampled in on_task_schedule,
        not much is done here.
        We return the list of milestones for this bracket in reverse
        (decreasing) order. The first entry is max_t, even if it is
        not a milestone in the bracket. This list contains the resource
        levels the task would reach if it ran to max_t without being stopped.

        :param task: Only task.task_id is used
        :return: See above
        """
        assert 'bracket' in kwargs
        bracket_id = kwargs['bracket']
        with HyperbandStopping_Manager.LOCK:
            bracket = self._brackets[bracket_id]
            self._task_info[task.task_id] = bracket_id
            levels = [x[0] for x in bracket._rungs]
            if levels[0] < self._max_t:
                levels.insert(0, self._max_t)
            return levels

    def _get_bracket(self, task_id):
        return self._brackets[self._task_info[task_id]]

    def on_task_report(self, task, result):
        """
        Decides whether task can continue or is to be stopped, and also
        whether the searcher should be updated (iff milestone is reached).
        If update_searcher = True and action = True, next_milestone is the
        next mileastone for the task (or None if there is none).

        :param task: Only task.task_id is used
        :param result: Current reported results from task
        :return: action, update_searcher, next_milestone
        """
        with HyperbandStopping_Manager.LOCK:
            action = False
            update_searcher = True
            next_milestone = None
            bracket_id = self._task_info[task.task_id]
            bracket = self._brackets[bracket_id]
            if 'done' not in result and result[self._time_attr] < self._max_t:
                action, update_searcher, next_milestone = bracket.on_result(
                    task, result[self._time_attr], result[self._reward_attr])
                # Special case: If config just reached the last milestone in
                # the bracket and survived, next_milestone is equal to max_t
                if action and update_searcher and (next_milestone is None):
                    next_milestone = self._max_t
            if action == False:
                self._num_stopped += 1
            return action, update_searcher, next_milestone, bracket_id, None

    def on_task_complete(self, task, result):
        with HyperbandStopping_Manager.LOCK:
            bracket = self._get_bracket(task.task_id)
            bracket.on_result(task, result[self._time_attr],
                              result[self._reward_attr])
            del self._task_info[task.task_id]

    def on_task_remove(self, task):
        with HyperbandStopping_Manager.LOCK:
            del self._task_info[task.task_id]

    def _sample_bracket(self):
        # Brackets are sampled in proportion to the number of configs started
        # in synchronous Hyperband in each bracket
        num_brackets = len(self._brackets)
        if num_brackets > 1:
            smax_plus1 = len(self._brackets[0]._rungs)
            rf = self._reduction_factor
            probs = np.array([
                (smax_plus1 / (smax_plus1 - s)) * (rf ** (smax_plus1 - s - 1))
                for s in range(num_brackets)])
            normalized = probs / probs.sum()
            return np.random.choice(num_brackets, p=normalized)
        else:
            return 0

    def on_task_schedule(self):
        with HyperbandStopping_Manager.LOCK:
            # Sample bracket for task to be scheduled
            bracket_id = self._sample_bracket()
        extra_kwargs = dict()
        extra_kwargs['bracket'] = bracket_id
        # First milestone the new config will get to:
        extra_kwargs['milestone'] = \
            self._brackets[bracket_id].get_first_milestone()
        return None, extra_kwargs

    def __repr__(self):
        reprstr = self.__class__.__name__ + '(' + \
                  'reward_attr: ' + self._reward_attr + \
                  ', time_attr: ' + self._time_attr + \
                  ', reduction_factor: ' + str(self._reduction_factor) + \
                  ', max_t: ' + str(self._max_t) + \
                  ', brackets: ' + str(self._brackets) + \
                  ')'
        return reprstr


class _Bracket(object):
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
        self._rungs = [(min_t * self.rf ** (k + s), dict())
                       for k in reversed(range(MAX_RUNGS))]

    def cutoff(self, recorded):
        if not recorded:
            return None
        return np.percentile(list(recorded.values()), (1 - 1 / self.rf) * 100)

    def on_result(self, task, cur_iter, cur_rew):
        """
        Decision on whether task may continue (action = True), or should be
        stoppped (action = False).
        milestone_reached is a flag whether cur_iter coincides with a milestone.
        If True, next_milestone is the next milestone after cur_iter, or None
        if there is none.

        :param task: Only need task.task_id
        :param cur_iter: Current time_attr value of task
        :param cur_rew: Current reward_attr value of task
        :return: action, milestone_reached, next_milestone
        """
        action = True
        milestone_reached = False
        next_milestone = None
        for milestone, recorded in self._rungs:
            if not (cur_iter < milestone or task.task_id in recorded):
                # Note: It is important for model-based searchers that
                # milestones are reached exactly, not jumped over. In
                # particular, if a future milestone is reported via
                # register_pending, its reward value has to be passed
                # later on via update.
                assert cur_iter == milestone, \
                    "cur_iter = {} > {} = milestone. Make sure to report time attributes covering all milestones".format(
                        cur_iter, milestone)
                milestone_reached = True
                cutoff = self.cutoff(recorded)
                if cutoff is not None and cur_rew < cutoff:
                    action = False
                if cur_rew is None:
                    logger.warning("Reward attribute is None! Consider"
                                   " reporting using a different field.")
                else:
                    recorded[task.task_id] = cur_rew
                break
            next_milestone = milestone
        return action, milestone_reached, next_milestone

    def get_first_milestone(self):
        return self._rungs[-1][0]

    def __repr__(self):
        iters = " | ".join([
            "Iter {:.3f}: {}".format(milestone, self.cutoff(recorded))
            for milestone, recorded in self._rungs
        ])
        return "Bracket: " + iters
