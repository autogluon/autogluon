import logging
import heapq
import copy

from .hyperband_stopping import RungEntry

logger = logging.getLogger(__name__)


class PromotionRungSystem(object):
    """
    Implements both the promotion and stopping logic for an asynchronous
    variant of Hyperband, known as ASHA:

    https://arxiv.org/abs/1810.05934

    In ASHA, configs sit paused at milestones (rung levels) until they get
    promoted, which means that a free task picks up their evaluation until
    the next milestone.

    Different to `StoppingRungSystem`, reward data at rungs is associated
    with configs, not with tasks. To avoid having to use config as key, we
    maintain unique integers as config keys.

    The stopping rule is simple: Per task_id, we record the config key and
    the milestone the task should be stopped at (it may still continue there,
    if it directly passes the promotion test).

    We do not directly support pause & resume here, so that in general,
    the evaluation for a promoted config is started from scratch. However,
    see `HyperbandScheduler.add_task`: The evaluation function train_fn can
    access the level to resume from via args.scheduler.resume_from, so
    pause & resume can be implemented there.

    Note: Say that an evaluation is resumed from level resume_from. If the
    train_fn does not implement pause & resume, it needs to start training from
    scratch, in which case metrics are reported for every epoch, also those <
    resume_from. At least for some modes of fitting the searcher model to data,
    this would lead to duplicate target values for the same extended config
    (x, r), which we want to avoid. The solution is to maintain resume_from in
    the data for the terminator (see `PromotionRungSystem._running`). Given
    this, we can report in `on_task_report` that the current metric data should
    not be used for the searcher model (`ignore_data = True`), namely as long
    as the evaluation has not yet gone beyond level resume_from.
    """
    def __init__(self, rung_levels, promote_quantiles, max_t):
        self.max_t = max_t
        # The data entry in _rungs is a dict mapping config_key to
        # (reward_value, was_promoted)
        assert len(rung_levels) == len(promote_quantiles)
        self._rungs = [
            RungEntry(level=x, prom_quant=y, data=dict())
            for x, y in reversed(list(zip(rung_levels, promote_quantiles)))]
        # Note: config_key are positions into _config, cast to str
        self._config = list()
        # _running maps str(task_id) to
        #   dict(config_key, milestone, resume_from),
        # which means task task_id runs evaluation of config_key until
        # time_attr reaches milestone. The resume_from field can be None. If
        # not, the task is running a config which has been promoted from
        # rung level resume_from. This info is required for on_result to
        # properly report ignore_data.
        self._running = dict()

    @staticmethod
    def _find_promotable_config(recorded, prom_quant, config_key=None):
        """
        Scans the top prom_quant fraction of recorded (sorted w.r.t. reward
        value) for config not yet promoted. If config_key is given, the key
        must also be equal to config_key.

        :param recorded: Dict to scan
        :param prom_quant: Quantile for promotion
        :param config_key: See above
        :return: Key of config if found, otherwise None
        """
        num_recorded = len(recorded)
        ret_key = None
        if num_recorded >= int(round(1.0 / prom_quant)):
            # Search for not yet promoted config in the top
            # prom_quant fraction
            def filter_pred(k, v):
                return (not v[1]) and (config_key is None or k == config_key)

            num_top = int(num_recorded * prom_quant)
            top_list = heapq.nlargest(
                num_top, recorded.items(), key=lambda x: x[1][0])
            try:
                ret_key = next(k for k, v in top_list if filter_pred(k, v))
            except StopIteration:
                ret_key = None
        return ret_key

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
        for rung in self._rungs:
            _milestone = rung.level
            prom_quant = rung.prom_quant
            _recorded = rung.data
            config_key = None
            if _milestone < self.max_t:
                config_key = self._find_promotable_config(
                    _recorded, prom_quant)
            if config_key is not None:
                recorded = _recorded
                milestone = _milestone
                break
            next_milestone = _milestone

        if config_key is None:
            # No promotable config in any rung
            return dict()
        else:
            # Mark config as promoted
            reward = recorded[config_key][0]
            assert not recorded[config_key][1]  # Sanity check
            recorded[config_key] = (reward, True)
            return {
                'config': self._config[int(config_key)],
                'config_key': config_key,
                'milestone': milestone,
                'next_milestone': next_milestone}

    def on_task_add(self, task, skip_rungs, **kwargs):
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
            # First milestone
            # If skip_rungs > 0, the lowest rung levels are not
            # milestones
            milestone = self.get_first_milestone(skip_rungs)
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
        self._running[str(task.task_id)] = {
            'config_key': config_key,
            'milestone': milestone,
            'resume_from': resume_from}

    def on_task_report(self, task, cur_iter, cur_rew, skip_rungs):
        """
        Decision on whether task may continue (task_continues = True), or should be
        stopped (task_continues = False).
        milestone_reached is a flag whether cur_iter coincides with a milestone.
        If True, next_milestone is the next milestone after cur_iter, or None
        if there is none.

        :param task:
        :param cur_iter: Current time_attr value of task
        :param cur_rew: Current reward_attr value of task
        :return: dict(task_continues, milestone_reached, next_milestone, ignore_data)
        """
        assert cur_rew is not None, \
            "Reward attribute must be a numerical value, not None"
        task_key = str(task.task_id)
        task_continues = True
        milestone_reached = False
        next_milestone = None
        milestone = self._running[task_key]['milestone']
        if cur_iter >= milestone:
            assert cur_iter == milestone, \
                "cur_iter = {} > {} = milestone. Make sure to report time attributes covering all milestones".format(
                    cur_iter, milestone)
            task_continues = False
            milestone_reached = True
            config_key = self._running[task_key]['config_key']
            assert self._config[int(config_key)] == task.args['config']
            try:
                rung_pos = next(i for i, v in enumerate(self._rungs)
                                if v.level == milestone)
                # Register reward at rung level (as not promoted)
                prom_quant = self._rungs[rung_pos].prom_quant
                recorded = self._rungs[rung_pos].data
                recorded[config_key] = (cur_rew, False)
                next_milestone = self._rungs[rung_pos - 1].level \
                    if rung_pos > 0 else self.max_t
                # Check whether config can be promoted immediately. If so,
                # we do not have to stop the task
                if milestone < self.max_t:
                    if self._find_promotable_config(
                            recorded, prom_quant,
                            config_key=config_key) is not None:
                        task_continues = True
                        recorded[config_key] = (cur_rew, True)
                        self._running[task_key] = {
                            'config_key': config_key,
                            'milestone': next_milestone,
                            'resume_from': None}
            except StopIteration:
                # milestone not a rung level. This can happen, in particular
                # if milestone == self.max_t
                pass
        resume_from = self._running[task_key]['resume_from']
        ignore_data = (resume_from is not None) and (cur_iter <= resume_from)
        return {
            'task_continues': task_continues,
            'milestone_reached': milestone_reached,
            'next_milestone': next_milestone,
            'ignore_data': ignore_data}

    def on_task_remove(self, task):
        del self._running[str(task.task_id)]

    def get_first_milestone(self, skip_rungs):
        return self._rungs[-(skip_rungs + 1)].level

    def get_milestones(self, skip_rungs):
        if skip_rungs > 0:
            milestone_rungs = self._rungs[:(-skip_rungs)]
        else:
            milestone_rungs = self._rungs
        return [x.level for x in milestone_rungs]

    def snapshot_rungs(self, skip_rungs):
        if skip_rungs > 0:
            _rungs = self._rungs[:(-skip_rungs)]
        else:
            _rungs = self._rungs
        return [(x.level, x.data) for x in _rungs]

    @staticmethod
    def _num_promotable_config(recorded, prom_quant):
        num_recorded = len(recorded)
        num_top = 0
        num_promotable = 0
        if num_recorded >= int(round(1.0 / prom_quant)):
            # Search for not yet promoted config in the top
            # prom_quant fraction
            num_top = int(num_recorded * prom_quant)
            top_list = heapq.nlargest(
                num_top, recorded.values(), key=lambda x: x[0])
            num_promotable = sum((not x) for _, x in top_list)
        return num_promotable, num_top

    def __repr__(self):
        iters = " | ".join([
            "Iter {:.3f}: {} of {}".format(
                r.level, *self._num_promotable_config(r.data, r.prom_quant))
            for r in self._rungs])
        return "Rung system: " + iters
