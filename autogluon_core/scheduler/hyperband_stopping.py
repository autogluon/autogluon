import logging
import numpy as np
import copy

logger = logging.getLogger(__name__)


def map_resource_to_index(resource, rf, min_t, max_t):
    max_rungs = int(np.log(max_t / min_t) / np.log(rf) + 1)
    index = int(np.round(np.log(resource * max_t / min_t) / np.log(rf)))
    index = max(min(index, max_rungs - 1), 0)
    return index


def _sample_bracket(num_brackets, max_num_rungs, rf):
    # Brackets are sampled in proportion to the number of configs started
    # in synchronous Hyperband in each bracket
    if num_brackets > 1:
        smax_plus1 = max_num_rungs
        probs = np.array([
            (smax_plus1 / (smax_plus1 - s)) * (rf ** (smax_plus1 - s - 1))
            for s in range(num_brackets)])
        normalized = probs / probs.sum()
        return np.random.choice(num_brackets, p=normalized)
    else:
        return 0


class HyperbandStopping_Manager(object):
    """Hyperband Manager

    Implements stopping rule which uses the brackets and rung levels defined
    in Hyperband. The overall algorithm is NOT what is published as ASHA
    (see HyperbandPromotion_Manager), but rather something resembling the
    median rule.

    Args:
        time_attr: str
            See HyperbandScheduler.
        reward_attr: str
            See HyperbandScheduler.
        max_t: int
            See HyperbandScheduler.
        grace_period: int
            See HyperbandScheduler.
        reduction_factor: int
            See HyperbandScheduler.
        brackets: int
            See HyperbandScheduler.

    """
    def __init__(
            self, time_attr, reward_attr, max_t, grace_period,
            reduction_factor, brackets):
        self._reward_attr = reward_attr
        self._time_attr = time_attr
        self._reduction_factor = reduction_factor
        self._max_t = max_t
        self._min_t = grace_period
        # Maps str(task_id) -> bracket_id
        self._task_info = dict()
        self._num_stopped = 0
        self._brackets = []
        for s in range(brackets):
            bracket = StoppingBracket(
                grace_period, max_t, reduction_factor, s)
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
        bracket = self._brackets[bracket_id]
        self._task_info[str(task.task_id)] = bracket_id
        levels = [x[0] for x in bracket._rungs]
        if levels[0] < self._max_t:
            levels.insert(0, self._max_t)
        return levels

    def _get_bracket(self, task_id):
        bracket_id = self._task_info[str(task_id)]
        return self._brackets[bracket_id], bracket_id

    def on_task_report(self, task, result):
        """
        This method is called by the reporter thread whenever a new metric
        value is received. It returns a dictionary with all the information
        needed for making decisions (e.g., stop / continue task, update
        model, etc)
        - task_continues: Should task continue or stop/pause?
        - update_searcher: True if rung level (or max_t) is hit, at which point
          the searcher should be updated
        - next_milestone: If hit rung level < max_t, this is the subsequent
          rung level (otherwise: None). Used for pending candidates
        - bracket_id: Bracket in which the task is running

        :param task: Only task.task_id is used
        :param result: Current reported results from task
        :return: See above
        """
        action = False
        update_searcher = True
        next_milestone = None
        bracket, bracket_id = self._get_bracket(task.task_id)
        if result[self._time_attr] < self._max_t:
            action, update_searcher, next_milestone = bracket.on_result(
                task, result[self._time_attr], result[self._reward_attr])
            # Special case: If config just reached the last milestone in
            # the bracket and survived, next_milestone is equal to max_t
            if action and update_searcher and (next_milestone is None):
                next_milestone = self._max_t
        if action == False:
            self._num_stopped += 1
        return {
            'task_continues': action,
            'update_searcher': update_searcher,
            'next_milestone': next_milestone,
            'bracket_id': bracket_id}

    def on_task_complete(self, task, result):
        bracket, _ = self._get_bracket(task.task_id)
        bracket.on_result(
            task, result[self._time_attr], result[self._reward_attr])
        self.on_task_remove(task)

    def on_task_remove(self, task):
        del self._task_info[str(task.task_id)]

    def _sample_bracket(self):
        return _sample_bracket(
            num_brackets=len(self._brackets),
            max_num_rungs=len(self._brackets[0]._rungs),
            rf=self._reduction_factor)

    def on_task_schedule(self):
        # Sample bracket for task to be scheduled
        bracket_id = self._sample_bracket()
        # 'milestone' is first milestone the new config will get to
        extra_kwargs = {
            'bracket': bracket_id,
            'milestone': self._brackets[bracket_id].get_first_milestone()}
        return None, extra_kwargs

    def snapshot_rungs(self, bracket_id):
        return self._brackets[bracket_id].snapshot_rungs()

    def resource_to_index(self, resource):
        return map_resource_to_index(
            resource, self._reduction_factor, self._min_t, self._max_t)

    def __repr__(self):
        reprstr = self.__class__.__name__ + '(' + \
                  'reward_attr: ' + self._reward_attr + \
                  ', time_attr: ' + self._time_attr + \
                  ', reduction_factor: ' + str(self._reduction_factor) + \
                  ', max_t: ' + str(self._max_t) + \
                  ', brackets: ' + str(self._brackets) + \
                  ')'
        return reprstr


class StoppingBracket(object):
    """Bookkeeping system to track the cutoffs.
    Rungs are created in reversed order so that we can more easily find
    the correct rung corresponding to the current iteration of the result.
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
        assert cur_rew is not None, \
            "Reward attribute must be a numerical value, not None"
        action = True
        milestone_reached = False
        next_milestone = None
        task_key = str(task.task_id)
        for milestone, recorded in self._rungs:
            if not (cur_iter < milestone or task_key in recorded):
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
                recorded[task_key] = cur_rew
                break
            next_milestone = milestone
        return action, milestone_reached, next_milestone

    def get_first_milestone(self):
        return self._rungs[-1][0]

    def snapshot_rungs(self):
        return [(x[0], copy.copy(x[1])) for x in self._rungs]

    def __repr__(self):
        iters = " | ".join([
            "Iter {:.3f}: {}".format(milestone, self.cutoff(recorded))
            for milestone, recorded in self._rungs
        ])
        return "Bracket: " + iters
