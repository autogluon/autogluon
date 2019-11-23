import logging
import numpy as np
import multiprocessing as mp
import heapq
import copy

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
            grace period, all share max_t and reduction_factor.
            If brackets == 1, we just run successive halving, for
            brackets > 1, we run Hyperband.
        keep_size_ratios (bool): If True,
            promotions are done only if the (current estimate of the) size ratio
            between rung and next rung are 1 / reduction_factor or better. This
            avoids higher rungs to get more populated than they would be in
            synchronous Hyperband. A drawback is that promotions to higher rungs
            take longer.
    """
    LOCK = mp.Lock()

    def __init__(
            self, time_attr, reward_attr, max_t, grace_period,
            reduction_factor, brackets, keep_size_ratios):
        # attr
        self._reward_attr = reward_attr
        self._time_attr = time_attr
        # hyperband params
        self._task_info = {}  # task_id -> bracket_id
        self._reduction_factor = reduction_factor
        self._max_t = max_t
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
        with HyperbandPromotion_Manager.LOCK:
            bracket = self._brackets[bracket_id]
            bracket.on_task_add(task, **kwargs)
            self._task_info[task.task_id] = bracket_id
            levels = [x[0] for x in bracket._rungs]
            if levels[0] < self._max_t:
                levels.insert(0, self._max_t)
            return levels

    def _get_bracket(self, task_id):
        return self._brackets[self._task_info[task_id]]

    def on_task_report(self, task, result):
        with HyperbandPromotion_Manager.LOCK:
            action = False
            update_searcher = True
            next_milestone = None
            bracket_id = self._task_info[task.task_id]
            bracket = self._brackets[bracket_id]
            rung_counts = bracket.get_rung_counts()
            if result[self._time_attr] < self._max_t:
                action, update_searcher, next_milestone = bracket.on_result(
                    task, result[self._time_attr], result[self._reward_attr])
            return action, update_searcher, next_milestone, bracket_id, rung_counts

    def on_task_complete(self, task, result):
        with HyperbandPromotion_Manager.LOCK:
            bracket = self._get_bracket(task.task_id)
            bracket.on_result(
                task, result[self._time_attr], result[self._reward_attr])
            bracket.on_task_remove(task)
            del self._task_info[task.task_id]

    def on_task_remove(self, task):
        with HyperbandPromotion_Manager.LOCK:
            self._get_bracket(task.task_id).on_task_remove(task)
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
        with HyperbandPromotion_Manager.LOCK:
            # Sample bracket for task to be scheduled
            bracket_id = self._sample_bracket()
            extra_kwargs = dict()
            extra_kwargs['bracket'] = bracket_id
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
    Different to _Bracket in hyperband_stopping, reward data at rungs is
    associated with configs, not with tasks. To avoid having to use config
    as key, we maintain unique integers as config keys.

    The stopping rule is simple: Per task_id, we record the config key and
    the milestone the task should be stopped at (it may still continue there,
    if it directly passes the promotion test).

    """
    def __init__(self, min_t, max_t, reduction_factor, s, keep_size_ratios):
        self.rf = reduction_factor
        self.max_t = max_t
        self.keep_size_ratios = keep_size_ratios
        MAX_RUNGS = int(np.log(max_t / min_t) / np.log(self.rf) - s + 1)
        # The second entry in each tuple in _rungs is a dict mapping
        # config_key to (reward_value, was_promoted)
        self._rungs = [(min_t * self.rf ** (k + s), dict())
                       for k in reversed(range(MAX_RUNGS))]
        # Note: config_key are positions into _config
        self._config = list()
        # _running maps task_id to tuples (config_key, milestone), which means
        # task task_id runs evaluation of config_key until time_attr reaches
        # milestone
        self._running = dict()
        # _count_tasks[m] counts the number of tasks started with target
        # milestone m. Here, m includes max_t. Used to implement the
        # keep_size_ratios rule
        self._count_tasks = dict()
        for milestone, _ in self._rungs:
            self._count_tasks[milestone] = 0
        if self._rungs and max_t > self._rungs[0][0]:
            self._count_tasks[max_t] = 0

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
            count_this = self._count_tasks[milestone]
            count_next = self._count_tasks[next_milestone]
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
            return self._config[config_key], config_key, milestone, next_milestone

    def on_task_add(self, task, **kwargs):
        """
        Called when new task is started. Depending on kwargs['new_config'], this
        could start an evaluation (True) or promote an existing config to the
        next milestone (False). In the latter case, kwargs contains additional
        information about the promotion.
        """
        new_config = kwargs.get('new_config', True)
        if new_config:
            # New config
            config_key = len(self._config)
            self._config.append(copy.copy(task.args['config']))
            # First milestone:
            milestone = self._rungs[-1][0]
        else:
            # Existing config is promoted
            # Note that self._rungs has already been updated in on_task_schedule
            assert 'milestone' in kwargs
            assert 'config_key' in kwargs
            config_key = kwargs['config_key']
            assert self._config[config_key] == task.args['config']
            milestone = kwargs['milestone']
        self._running[task.task_id] = (config_key, milestone)
        self._count_tasks[milestone] += 1

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
        :return: action, milestone_reached, next_milestone
        """
        action = True
        milestone_reached = False
        next_milestone = None
        milestone = self._running[task.task_id][1]
        if cur_iter >= milestone:
            assert cur_iter == milestone, \
                "cur_iter = {} > {} = milestone. Make sure to report time attributes covering all milestones".format(
                    cur_iter, milestone)
            action = False
            milestone_reached = True
            config_key = self._running[task.task_id][0]
            assert self._config[config_key] == task.args['config']
            if cur_rew is None:
                logger.warning(
                    "Reward attribute is None! Consider reporting using "
                    "a different field.")
            else:
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
                            self._running[task.task_id] = (config_key, next_milestone)
                            self._count_tasks[next_milestone] += 1
                except StopIteration:
                    # milestone not a rung level. This can happen, in particular
                    # if milestone == self.max_t
                    pass
        return action, milestone_reached, next_milestone

    def get_rung_counts(self):
        return self._count_tasks

    def on_task_remove(self, task):
        del self._running[task.task_id]

    def get_first_milestone(self):
        return self._rungs[-1][0]

    def _num_promotable_config(self, recorded):
        num_recorded = len(recorded)
        num_top = 0
        num_promotable = 0
        if num_recorded >= self.rf:
            # Search for not yet promoted config in the top
            # 1 / self.rf fraction
            num_top = int(num_recorded / self.rf)
            top_list = heapq.nlargest(
                num_top, recorded.items(), key=lambda x: x[1][0])
            num_promotable = sum([(not v[1]) for v in top_list.values()])
        return num_promotable, num_top

    def __repr__(self):
        iters = " | ".join([
            "Iter {:.3f}: {} of {}".format(
                milestone, *self._num_promotable_config(recorded))
            for milestone, recorded in self._rungs
        ])
        return "Bracket: " + iters
