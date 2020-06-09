import logging
import numpy as np
import heapq
import copy

from .hyperband_stopping import map_resource_to_index, _sample_bracket

logger = logging.getLogger(__name__)


class HyperbandPromotion_Manager(object):
    """Hyperband Manager
    
    Implements both the promotion and stopping logic for an asynchronous
    variant of Hyperband, known as ASHA:
    https://arxiv.org/abs/1810.05934

    In ASHA, configs sit paused at milestones (rung levels) in their
    bracket, until they get promoted, which means that a free task picks
    up their evaluation until the next milestone.

    We do not directly support pause & resume here, so that in general,
    the evaluation for a promoted config is started from scratch. However,
    see Hyperband_Scheduler.add_task, task.args['resume_from']: the
    evaluation function receives info about the promotion, so pause &
    resume can be implemented there.

    Note: If the evaluation function does not implement pause & resume, it
    needs to start training from scratch, in which case metrics are reported
    for every epoch, also those < task.args['resume_from']. At least for some
    modes of fitting the searcher model to data, this would lead to duplicate
    target values for the same extended config (x, r), which we want to avoid.
    The solution is to maintain task.args['resume_from'] in the data for the
    terminator (see PromotionBracket._running). Given this, we can report
    in on_task_report that the current metric data should not be used for the
    searcher model (ignore_data = True), namely as long as the evaluation has
    not yet gone beyond level task.args['resume_from'].

    Args:
        time_attr : str
            See HyperbandScheduler.
        reward_attr : str
            See HyperbandScheduler.
        max_t : int
            See HyperbandScheduler.
        grace_period : int
            See HyperbandScheduler.
        reduction_factor : int
            See HyperbandScheduler.
        brackets : int
            See HyperbandScheduler.
        keep_size_ratios : bool
            See HyperbandScheduler.
    """
    def __init__(
            self, time_attr, reward_attr, max_t, grace_period,
            reduction_factor, brackets, keep_size_ratios):
        self._reward_attr = reward_attr
        self._time_attr = time_attr
        self._reduction_factor = reduction_factor
        self._max_t = max_t
        self._min_t = grace_period
        # Maps str(task_id) -> bracket_id
        self._task_info = dict()
        self._brackets = []
        for s in range(brackets):
            bracket = PromotionBracket(
                grace_period, max_t, reduction_factor, s, keep_size_ratios)
            if not bracket._rungs:
                break
            self._brackets.append(bracket)

    def on_task_add(self, task, **kwargs):
        assert 'bracket' in kwargs
        bracket_id = kwargs['bracket']
        bracket = self._brackets[bracket_id]
        bracket.on_task_add(task, **kwargs)
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
        See docstring of HyperbandStopping_Manager. Additional entries:
        - rung_counts: Occupancy counts per rung level
        - ignore_data: If True, the metric value should not be added to the
          dataset of the searcher

        ignore_data = True iff the task is running a config which has been
        promoted, and the resource level is <= the rung from where the config
        was promoted. This happens if the evaluation function does not support
        pause&resume and has to be started from scratch.
        """
        action = False
        update_searcher = True
        next_milestone = None
        ignore_data = False
        bracket, bracket_id = self._get_bracket(task.task_id)
        rung_counts = bracket.get_rung_counts()
        if result[self._time_attr] < self._max_t:
            action, update_searcher, next_milestone, ignore_data = \
                bracket.on_result(task, result[self._time_attr],
                                  result[self._reward_attr])
        return {
            'task_continues': action,
            'update_searcher': update_searcher,
            'next_milestone': next_milestone,
            'bracket_id': bracket_id,
            'rung_counts': rung_counts,
            'ignore_data': ignore_data}

    def on_task_complete(self, task, result):
        bracket, _ = self._get_bracket(task.task_id)
        bracket.on_result(
            task, result[self._time_attr], result[self._reward_attr])
        self.on_task_remove(task)

    def on_task_remove(self, task):
        task_id = task.task_id
        bracket, _ = self._get_bracket(task_id)
        bracket.on_task_remove(task)
        del self._task_info[str(task_id)]

    def _sample_bracket(self):
        return _sample_bracket(
            num_brackets=len(self._brackets),
            max_num_rungs=len(self._brackets[0]._rungs),
            rf=self._reduction_factor)

    def on_task_schedule(self):
        # Sample bracket for task to be scheduled
        bracket_id = self._sample_bracket()
        extra_kwargs = {'bracket': bracket_id}
        bracket = self._brackets[bracket_id]
        # Check whether config can be promoted in that bracket
        config, config_key, milestone, next_milestone = \
            bracket.on_task_schedule()
        if config is not None:
            extra_kwargs['milestone'] = next_milestone
            extra_kwargs['config_key'] = config_key
            extra_kwargs['resume_from'] = milestone
        else:
            # First milestone the new config will get to
            extra_kwargs['milestone'] = bracket.get_first_milestone()
        return config, extra_kwargs

    def resource_to_index(self, resource):
        return map_resource_to_index(
            resource, self._reduction_factor, self._min_t, self._max_t)

    def snapshot_rungs(self, bracket_id):
        return self._brackets[bracket_id].snapshot_rungs()

    def __repr__(self):
        reprstr = self.__class__.__name__ + '(' + \
                  'reward_attr: ' + self._reward_attr + \
                  ', time_attr: ' + self._time_attr + \
                  ', reduction_factor: ' + str(self._reduction_factor) + \
                  ', max_t: ' + str(self._max_t) + \
                  ', brackets: ' + str(self._brackets) + \
                  ')'
        return reprstr


class PromotionBracket(object):
    """
    Different to StoppingBracket in hyperband_stopping, reward data at rungs is
    associated with configs, not with tasks. To avoid having to use config
    as key, we maintain unique integers as config keys.

    The stopping rule is simple: Per task_id, we record the config key and
    the milestone the task should be stopped at (it may still continue there,
    if it directly passes the promotion test).
    """
    def __init__(
            self, min_t, max_t, reduction_factor, s, keep_size_ratios):
        self.rf = reduction_factor
        self.max_t = max_t
        self.keep_size_ratios = keep_size_ratios
        MAX_RUNGS = int(np.log(max_t / min_t) / np.log(self.rf) - s + 1)
        # The second entry in each tuple in _rungs is a dict mapping
        # config_key to (reward_value, was_promoted)
        self._rungs = [(min_t * self.rf ** (k + s), dict())
                       for k in reversed(range(MAX_RUNGS))]
        # Note: config_key are positions into _config, cast to str
        self._config = list()
        # _running maps str(task_id) to tuples
        #   (config_key, milestone, resume_from),
        # which means task task_id runs evaluation of config_key until
        # time_attr reaches milestone. The resume_from field can be None. If
        # not, the task is running a config which has been promoted from
        # rung level resume_from. This info is required for on_result to
        # properly report ignore_data.
        self._running = dict()
        # _count_tasks[m] counts the number of tasks started with target
        # milestone m. Here, m includes max_t. Used to implement the
        # keep_size_ratios rule
        self._count_tasks = dict()
        for milestone, _ in self._rungs:
            self._count_tasks[str(milestone)] = 0
        if self._rungs and max_t > self._rungs[0][0]:
            self._count_tasks[str(max_t)] = 0

    def _find_promotable_config(self, recorded, config_key=None):
        """
        Scans the top (1 / self.rf) fraction of recorded (sorted w.r.t. reward
        value) for config not yet promoted. If config_key is given, the key
        must also be equal to config_key.

        Note: It would be more efficient to keep the dictionary as a heap,
        instead of rebuilding it each time.

        :param recorded: Dict to scan
        :param config_key: See above
        :return: Key of config if found, otherwise None
        """
        num_recorded = len(recorded)
        ret_key = None
        if num_recorded >= self.rf:
            # Search for not yet promoted config in the top
            # 1 / self.rf fraction
            def filter_pred(k, v):
                return (not v[1]) and (config_key is None or k == config_key)

            num_top = int(num_recorded / self.rf)
            top_list = heapq.nlargest(
                num_top, recorded.items(), key=lambda x: x[1][0])
            try:
                ret_key = next(k for k, v in top_list if filter_pred(k, v))
            except StopIteration:
                ret_key = None
        return ret_key

    def _do_skip_promotion(self, milestone, next_milestone):
        skip_promotion = False
        if self.keep_size_ratios:
            count_this = self._count_tasks[str(milestone)]
            count_next = self._count_tasks[str(next_milestone)]
            # We skip the promotion if the ideal size ratio is currently
            # violated. This is a little more permissive than we could be.
            # Since a promotion increases count_next, we could also use
            #    (count_next + 1) * self.rf > count_this
            skip_promotion = (count_next * self.rf > count_this)
        return skip_promotion

    def on_task_schedule(self):
        """
        Used to implement _promote_config of scheduler. Searches through rungs
        to find a config which can be promoted. If one is found, we return the
        config and other info (config_key, current milestone, milestone to be
        promoted to).
        We also mark the config as being promoted at the rung level it sits
        right now.
        """
        config_key = None
        next_milestone = self.max_t
        milestone = None
        recorded = None
        for _milestone, _recorded in self._rungs:
            config_key = None
            if _milestone < self.max_t:
                skip_promotion = self._do_skip_promotion(
                    _milestone, next_milestone)
                config_key = self._find_promotable_config(_recorded) \
                    if not skip_promotion else None
            if config_key is not None:
                recorded = _recorded
                milestone = _milestone
                break
            next_milestone = _milestone

        if config_key is None:
            # No promotable config in any rung
            return None, None, None, None
        else:
            # Mark config as promoted
            reward = recorded[config_key][0]
            assert not recorded[config_key][1]
            recorded[config_key] = (reward, True)
            return self._config[int(config_key)], config_key, milestone, \
                   next_milestone

    def on_task_add(self, task, **kwargs):
        """
        Called when new task is started. Depending on kwargs['new_config'],
        this could start an evaluation (True) or promote an existing config
        to the next milestone (False). In the latter case, kwargs contains
        additional information about the promotion.
        """
        new_config = kwargs.get('new_config', True)
        if new_config:
            # New config
            config_key = str(len(self._config))
            self._config.append(copy.copy(task.args['config']))
            # First milestone:
            milestone = self._rungs[-1][0]
            resume_from = None
        else:
            # Existing config is promoted
            # Note that self._rungs has already been updated in
            # on_task_schedule
            assert 'milestone' in kwargs
            assert 'config_key' in kwargs
            config_key = kwargs['config_key']
            assert self._config[int(config_key)] == task.args['config']
            milestone = kwargs['milestone']
            resume_from = kwargs.get('resume_from')
        self._running[str(task.task_id)] = (config_key, milestone, resume_from)
        self._count_tasks[str(milestone)] += 1

    def on_result(self, task, cur_iter, cur_rew):
        """
        Decision on whether task may continue (action = True), or should be
        stopped (action = False).
        milestone_reached is a flag whether cur_iter coincides with a milestone.
        If True, next_milestone is the next milestone after cur_iter, or None
        if there is none.

        :param task: Only need task.task_id
        :param cur_iter: Current time_attr value of task
        :param cur_rew: Current reward_attr value of task
        :return: action, milestone_reached, next_milestone, ignore_data
        """
        assert cur_rew is not None, \
            "Reward attribute must be a numerical value, not None"
        task_key = str(task.task_id)
        action = True
        milestone_reached = False
        next_milestone = None
        milestone = self._running[task_key][1]
        if cur_iter >= milestone:
            assert cur_iter == milestone, \
                "cur_iter = {} > {} = milestone. Make sure to report time attributes covering all milestones".format(
                    cur_iter, milestone)
            action = False
            milestone_reached = True
            config_key = self._running[task_key][0]
            assert self._config[int(config_key)] == task.args['config']
            try:
                rung_pos = next(i for i, v in enumerate(self._rungs)
                                if v[0] == milestone)
                # Register reward at rung level (as not promoted)
                recorded = self._rungs[rung_pos][1]
                recorded[config_key] = (cur_rew, False)
                next_milestone = self._rungs[rung_pos - 1][0] \
                    if rung_pos > 0 else self.max_t
                # Check whether config can be promoted immediately. If so,
                # we do not have to stop the task
                if milestone < self.max_t:
                    skip_promotion = self._do_skip_promotion(
                        milestone, next_milestone)
                    if (not skip_promotion) and (self._find_promotable_config(
                            recorded, config_key=config_key) is not None):
                        action = True
                        recorded[config_key] = (cur_rew, True)
                        self._running[task_key] = (
                            config_key, next_milestone, None)
                        self._count_tasks[str(next_milestone)] += 1
            except StopIteration:
                # milestone not a rung level. This can happen, in particular
                # if milestone == self.max_t
                pass
        resume_from = self._running[task_key][2]
        ignore_data = (resume_from is not None) and (cur_iter <= resume_from)

        return action, milestone_reached, next_milestone, ignore_data

    def get_rung_counts(self):
        return self._count_tasks

    def on_task_remove(self, task):
        del self._running[str(task.task_id)]

    def get_first_milestone(self):
        return self._rungs[-1][0]

    def snapshot_rungs(self):
        return [(x[0], copy.copy(x[1])) for x in self._rungs]

    def _num_promotable_config(self, recorded):
        num_recorded = len(recorded)
        num_top = 0
        num_promotable = 0
        if num_recorded >= self.rf:
            # Search for not yet promoted config in the top
            # 1 / self.rf fraction
            num_top = int(num_recorded / self.rf)
            top_list = heapq.nlargest(
                num_top, recorded.values(), key=lambda x: x[0])
            num_promotable = sum((not x) for _, x in top_list)
        return num_promotable, num_top

    def __repr__(self):
        iters = " | ".join([
            "Iter {:.3f}: {} of {}".format(
                milestone, *self._num_promotable_config(recorded))
            for milestone, recorded in self._rungs
        ])
        return "Bracket: " + iters
